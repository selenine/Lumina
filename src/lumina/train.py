import contextlib
import dataclasses
import functools
import math
import os

import lpips
import torch
import wandb
from accelerate import Accelerator
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizerFast

from .configs import (
    DataConfig,
    DecoderConfig,
    DecoderTrainConfig,
    SamplerConfig,
    SamplerTrainConfig,
    latest_checkpoint,
)
from .data import Stream, collate, collate_pixels
from .nn.latents import Decoder, Discriminator, Encoder, PixelDiscriminator
from .nn.model import DiT, Sampler


def cosine_lr(step: int, n_warmup: int, max_steps: int) -> float:
    if step < n_warmup:
        return (step + 1) / max(n_warmup, 1)

    progress = (step - n_warmup) / max(max_steps - n_warmup, 1)

    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class _Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        cfg: SamplerTrainConfig | DecoderTrainConfig,
        data: DataConfig,
        collate_fn,
        tag: str,
    ) -> None:
        self.cfg = cfg
        self.data = data
        self.step = 0
        self.epoch = 0

        self.accel = Accelerator(
            mixed_precision=cfg.mixed_precision if torch.cuda.is_available() else "no",
            log_with="wandb" if cfg.wandb_project else None,
        )
        self.device = self.accel.device

        if cfg.wandb_project:
            self.accel.init_trackers(
                cfg.wandb_project,
                config={
                    tag: dataclasses.asdict(cfg),
                    "data": dataclasses.asdict(data),
                },
            )

        dataloader = DataLoader(
            Stream(data, cfg.seed),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            **(
                {"prefetch_factor": 4, "persistent_workers": True}
                if cfg.num_workers > 0
                else {}
            ),
        )

        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_lr(step, cfg.n_warmup, cfg.max_steps),
        )

        self.model, self.dataloader, self.optimizer, self.scheduler = (
            self.accel.prepare(model, dataloader, optimizer, scheduler)
        )

        self.net = self.accel.unwrap_model(self.model)
        self.ema = (
            AveragedModel(
                self.net,
                multi_avg_fn=get_ema_multi_avg_fn(cfg.ema_decay),
                use_buffers=True,
            )
            if cfg.ema_decay > 0
            else None
        )
        self.encoder = Encoder(data.encoder).to(self.device)

        if cfg.compile:
            self.model = torch.compile(self.model)

        os.makedirs(cfg.output_dir, exist_ok=True)

    def restore(
        self, resume: str | None = None, stats: str | None = None
    ) -> dict | None:
        if resume:
            state = torch.load(resume, map_location="cpu", weights_only=True)
            if "model" in state:
                self.net.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
                if self.ema is not None and "ema" in state:
                    self.ema.module.load_state_dict(state["ema"])
                    self.ema.n_averaged.fill_(1)
                if "stats" in state:
                    self.encoder.load_stats(state["stats"])
                self.step = state["step"]
            else:
                self.net.load_state_dict(state)
            self.accel.print(f"resumed from {resume} at step {self.step}")

            return state

        if stats:
            state = torch.load(stats, map_location="cpu", weights_only=True)
            if "stats" not in state:
                raise ValueError(f"{stats} holds no encoder statistics")
            self.encoder.load_stats(state["stats"])
            self.accel.print(f"took encoder statistics from {stats}")

        return None

    def fit_stats(self) -> None:
        if bool(self.encoder.pixel_fitted):
            return

        self.encoder.fit_stats(self.dataloader, self.data.n_stat_batches)
        pixel = [round(v, 4) for v in self.encoder.pixel_mean.flatten().tolist()]
        self.accel.print(
            f"pixel mean {pixel} latent std {self.encoder.latent_std.mean():.4f}"
        )

    @contextlib.contextmanager
    def averaged(self):
        if self.ema is None:
            yield self.net
            return

        backup = {
            key: value.detach().clone() for key, value in self.net.state_dict().items()
        }
        self.net.load_state_dict(self.ema.module.state_dict())
        try:
            yield self.net
        finally:
            self.net.load_state_dict(backup)

    def sample_dir(self) -> str:
        path = os.path.join(self.cfg.output_dir, "samples")
        os.makedirs(path, exist_ok=True)

        return path

    def tracker(self):
        if not self.cfg.wandb_project:
            return None

        with contextlib.suppress(Exception):
            return self.accel.get_tracker("wandb", unwrap=True)

        return None

    def save_checkpoint(self, extra: dict | None = None) -> None:
        self.accel.wait_for_everyone()
        if not self.accel.is_main_process:
            return

        state = {
            "step": self.step,
            "model": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "stats": self.encoder.stats_dict(),
        }
        if self.ema is not None:
            state["ema"] = self.ema.module.state_dict()
        if extra is not None:
            state.update(extra)

        self.accel.save(
            state, os.path.join(self.cfg.output_dir, f"checkpoint_{self.step:08d}.pt")
        )

    def save_final(self, name: str) -> None:
        self.accel.wait_for_everyone()
        if not self.accel.is_main_process:
            return

        weights = (
            self.ema.module.state_dict()
            if self.ema is not None
            else self.net.state_dict()
        )

        self.accel.save(
            {"model": weights, "stats": self.encoder.stats_dict()},
            os.path.join(self.cfg.output_dir, name),
        )

    def run(self, step_fn, sample_fn=None, checkpoint_extra=None) -> None:
        cfg = self.cfg

        pbar = tqdm(
            total=cfg.max_steps,
            initial=self.step,
            desc="step",
            disable=not self.accel.is_local_main_process,
        )

        while self.step < cfg.max_steps:
            self.model.train()
            total, seen = 0.0, 0

            for batch in self.dataloader:
                self.optimizer.zero_grad()
                loss, extra = step_fn(batch)
                self.accel.backward(loss)

                grad_norm = None
                if cfg.max_grad_norm > 0 and self.accel.sync_gradients:
                    grad_norm = self.accel.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    ).item()

                self.optimizer.step()
                self.scheduler.step()

                if self.ema is not None:
                    self.ema.update_parameters(self.net)

                value = loss.item()
                lr = self.scheduler.get_last_lr()[0]
                total += value
                seen += 1
                self.step += 1
                pbar.update(1)

                postfix = {"loss": f"{value:.4f}", "avg": f"{total / seen:.4f}"}
                postfix.update({key: f"{v:.4f}" for key, v in extra.items()})
                postfix["lr"] = f"{lr:.2e}"
                if grad_norm is not None:
                    postfix["gn"] = f"{grad_norm:.2f}"
                pbar.set_postfix(postfix, refresh=False)

                if cfg.log_interval > 0 and self.step % cfg.log_interval == 0:
                    metrics = {"loss": value, **extra, "lr": lr, "epoch": self.epoch}
                    if grad_norm is not None:
                        metrics["grad_norm"] = grad_norm
                    self.accel.log(metrics, step=self.step)

                if cfg.save_interval > 0 and self.step % cfg.save_interval == 0:
                    self.save_checkpoint(
                        checkpoint_extra() if checkpoint_extra is not None else None
                    )

                if (
                    sample_fn is not None
                    and cfg.sample_interval > 0
                    and self.step % cfg.sample_interval == 0
                ):
                    sample_fn()

                if self.step >= cfg.max_steps:
                    break

            self.accel.log({"epoch_loss": total / max(seen, 1)}, step=self.step)
            self.epoch += 1

        pbar.close()


def train_sampler(
    model: DiT,
    cfg: SamplerTrainConfig,
    data: DataConfig,
    sampler: SamplerConfig | None = None,
    decoder: DecoderConfig | None = None,
    resume: str | None = None,
):
    tokenizer = CLIPTokenizerFast.from_pretrained(data.clip)

    runner = _Runner(
        model,
        cfg,
        data,
        functools.partial(collate, tokenizer=tokenizer, max_tokens=data.max_tokens),
        "sampler",
    )
    accel, device = runner.accel, runner.device

    text_encoder = (
        CLIPTextModel.from_pretrained(data.clip).to(device).eval().requires_grad_(False)
    )

    runner.restore(resume=resume)
    runner.fit_stats()

    def step_fn(batch):
        pixels, input_ids, attention_mask = batch

        with torch.no_grad():
            x = runner.encoder.encode(pixels)
            c = text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

        def adaptive_l2(error: torch.Tensor) -> torch.Tensor:
            d = error.float().pow(2).flatten(1).mean(-1)
            w = (d.detach() + 1e-3).pow(-0.5)

            return (w * d).mean()

        def sample_r_t(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
            s = (
                torch.randn(batch, 2, device=device) * cfg.p_std + cfg.p_mean
            ).sigmoid()
            r, t = s.amin(dim=-1), s.amax(dim=-1)

            return torch.where(torch.rand(batch, device=device) >= cfg.p_ratio, t, r), t

        b = x.shape[0]
        r, t = sample_r_t(b)
        rb = r.view(-1, *([1] * (x.dim() - 1)))
        tb = t.view(-1, *([1] * (x.dim() - 1)))

        drop = torch.rand(b, device=device) < cfg.p_uncond

        eps = torch.randn_like(x)
        z = (1.0 - tb) * x + tb * eps
        v = eps - x

        u, dudt = torch.func.jvp(  # ty: ignore
            lambda z, r, t: runner.model(z, c, r, t, drop, attention_mask),
            (z, r, t),
            (v, torch.zeros_like(r), torch.ones_like(t)),
        )

        tgt = (v - (tb - rb) * dudt).detach()

        return adaptive_l2(u - tgt), {}

    pipeline = None

    can_sample = (
        bool(cfg.sample_prompts)
        and sampler is not None
        and decoder is not None
        and bool(latest_checkpoint(decoder.train.output_dir, "decoder.pt"))
    )

    if cfg.sample_interval > 0 and not can_sample:
        accel.print(
            "sampling disabled: no decoder in "
            f"{decoder.train.output_dir if decoder is not None else 'decoder.train'}"
        )

    def log_samples() -> None:
        nonlocal pipeline

        accel.wait_for_everyone()
        if not accel.is_main_process:
            return

        if pipeline is None:
            pipeline = Sampler(
                sampler,
                decoder,
                device=device,
                dit=runner.net,
                stats=runner.encoder.stats_dict(),
            )

        out_dir = runner.sample_dir()

        with runner.averaged():
            images = [
                pipeline.generate(
                    prompt,
                    num_images=1,
                    num_steps=cfg.sample_steps,
                    guidance=cfg.sample_guidance,
                )[0]
                for prompt in cfg.sample_prompts
            ]

        paths = []
        for i, image in enumerate(images):
            path = os.path.join(out_dir, f"step_{runner.step:08d}_{i:02d}.png")
            save_image(image.float(), path)
            paths.append(path)

        tracker = runner.tracker()
        if tracker is not None:
            tracker.log(
                {
                    "samples": [
                        wandb.Image(path, caption=prompt)
                        for path, prompt in zip(paths, cfg.sample_prompts)
                    ]
                },
                step=runner.step,
            )

    runner.run(step_fn, log_samples if can_sample else None)
    runner.save_final("model.pt")

    accel.end_training()


def train_decoder(
    decoder: Decoder,
    cfg: DecoderTrainConfig,
    data: DataConfig,
    resume: str | None = None,
    stats: str | None = None,
):
    runner = _Runner(decoder, cfg, data, collate_pixels, "decoder")
    accel, device = runner.accel, runner.device

    perceptual = (
        lpips.LPIPS(net=cfg.lpips_net).to(device).eval().requires_grad_(False)
        if cfg.lpips_weight > 0
        else None
    )

    if cfg.gan_backbone not in ("dino", "pixel"):
        raise ValueError(
            f"unknown gan_backbone {cfg.gan_backbone!r}, expected 'dino' or 'pixel'"
        )

    latent_gan = cfg.gan_backbone == "dino"

    disc = disc_opt = disc_sched = None
    if cfg.gan_weight > 0:
        disc = (
            Discriminator(runner.encoder.d_latent, cfg.gan_channels, cfg.gan_layers)
            if latent_gan
            else PixelDiscriminator(cfg.gan_channels, cfg.gan_layers)
        )
        disc_opt = optim.AdamW(disc.parameters(), lr=cfg.gan_lr, betas=(0.5, 0.9))
        disc_sched = LambdaLR(
            disc_opt,
            lr_lambda=lambda step: cosine_lr(
                step, cfg.n_warmup, max(cfg.max_steps - cfg.gan_start, 1)
            ),
        )
        disc, disc_opt, disc_sched = accel.prepare(disc, disc_opt, disc_sched)

    state = runner.restore(resume=resume, stats=stats)
    if state is not None and disc is not None and "disc" in state:
        accel.unwrap_model(disc).load_state_dict(state["disc"])
        disc_opt.load_state_dict(state["disc_optimizer"])
        disc_sched.load_state_dict(state["disc_scheduler"])

    runner.fit_stats()

    preview = None

    def step_fn(batch):
        nonlocal preview

        (pixels,) = batch
        if preview is None:
            preview = pixels[: cfg.n_samples].clone()

        with torch.no_grad():
            latent = runner.encoder.encode(pixels, normalize=False)

        recon = runner.model(latent)

        l1 = (recon - pixels).abs().mean()
        loss = cfg.l1_weight * l1
        metrics = {"l1": l1.item()}

        if perceptual is not None:
            perc = perceptual(2.0 * recon - 1.0, 2.0 * pixels - 1.0).mean()
            loss = loss + cfg.lpips_weight * perc
            metrics["lpips"] = perc.item()

        if disc is not None and runner.step >= cfg.gan_start:
            if latent_gan:
                truth = runner.encoder.normalize(latent)
                forged = runner.encoder.features(recon)
            else:
                truth, forged = pixels, recon

            disc_opt.zero_grad()
            real = F.relu(1.0 - disc(truth)).mean()
            fake = F.relu(1.0 + disc(forged.detach())).mean()
            d_loss = 0.5 * (real + fake)
            accel.backward(d_loss)
            disc_opt.step()
            disc_sched.step()

            adv = -disc(forged).mean()
            loss = loss + cfg.gan_weight * adv
            metrics["adv"] = adv.item()
            metrics["disc"] = d_loss.item()

        return loss, metrics

    def checkpoint_extra() -> dict | None:
        if disc is None:
            return None

        return {
            "disc": accel.unwrap_model(disc).state_dict(),
            "disc_optimizer": disc_opt.state_dict(),
            "disc_scheduler": disc_sched.state_dict(),
        }

    def log_recon() -> None:
        accel.wait_for_everyone()
        if not accel.is_main_process or preview is None:
            return

        with runner.averaged(), torch.no_grad():
            recon = runner.net(runner.encoder.encode(preview, normalize=False))

        grid = make_grid(
            torch.cat([preview, recon.float().clamp(0, 1)]).cpu(),
            nrow=preview.shape[0],
        )
        path = os.path.join(runner.sample_dir(), f"step_{runner.step:08d}.png")
        save_image(grid, path)

        tracker = runner.tracker()
        if tracker is not None:
            tracker.log(
                {"recon": wandb.Image(path, caption="source over reconstruction")},
                step=runner.step,
            )

    runner.run(step_fn, log_recon, checkpoint_extra)
    runner.save_final("decoder.pt")

    accel.end_training()
