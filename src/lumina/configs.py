from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args

import yaml


@dataclass
class DiTConfig:
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    d_mlp: int = 2048
    n_layers: int = 6


@dataclass
class SamplerTrainConfig:
    p_mean: float = -0.4
    p_std: float = 1.0
    p_ratio: float = 0.25
    p_uncond: float = 0.1
    n_warmup: int = 1000
    max_steps: int = 200_000
    batch_size: int = 128
    num_workers: int = 0
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    ema_decay: float = 0.9999
    mixed_precision: str = "bf16"
    compile: bool = True
    log_interval: int = 50
    wandb_project: str = ""
    save_interval: int = 5000
    sample_interval: int = 2500
    sample_prompts: list[str] = field(default_factory=list)
    sample_steps: int = 2
    sample_guidance: float = 3.0
    output_dir: str = "./checkpoints"


@dataclass
class DecoderTrainConfig:
    n_warmup: int = 1000
    max_steps: int = 100_000
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 42
    lr: float = 1e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    l1_weight: float = 1.0
    lpips_weight: float = 1.0
    lpips_net: str = "vgg"
    gan_weight: float = 0.1
    gan_start: int = 20_000
    gan_backbone: str = "dino"
    gan_lr: float = 2e-4
    gan_channels: int = 64
    gan_layers: int = 3
    mixed_precision: str = "bf16"
    compile: bool = True
    log_interval: int = 50
    wandb_project: str = ""
    save_interval: int = 5000
    sample_interval: int = 2500
    n_samples: int = 4
    output_dir: str = "./decoder"


@dataclass
class DecoderConfig:
    resolution: int = 256
    d_latent: int = 768
    d_model: int = 768
    n_heads: int = 12
    d_head: int = 64
    d_mlp: int = 3072
    n_layers: int = 12

    train: DecoderTrainConfig = field(default_factory=DecoderTrainConfig)


@dataclass
class SamplerConfig:
    img_size: int = 16
    patch_size: int = 1
    d_caption: int = 768
    n_channels: int = 768
    clip: str = "openai/clip-vit-large-patch14"

    dit: DiTConfig = field(default_factory=DiTConfig)
    train: SamplerTrainConfig = field(default_factory=SamplerTrainConfig)


@dataclass
class DataConfig:
    dataset: str = "Fhrozen/relaion-art"
    split: str = "train"
    resolution: int = 256
    preprocessed: bool = False
    local: bool = False
    shuffle: bool = True
    shuffle_buffer: int = 10_000
    image_key: str = "image"
    caption_key: str = "text"
    max_tokens: int = 77
    encoder: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    n_stat_batches: int = 50
    clip: str = "openai/clip-vit-large-patch14"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


def latest_checkpoint(output_dir: str, name: str = "model.pt") -> str:
    final = Path(output_dir) / name
    if final.exists():
        return str(final)

    checkpoints = sorted(Path(output_dir).glob("checkpoint_*.pt"))

    return str(checkpoints[-1]) if checkpoints else ""


def _build(cls: type, raw: Any, path: str) -> Any:
    known = {f.name: f for f in fields(cls)}

    kwargs = {}
    for name, value in raw.items():
        f = known[name]
        child = f"{path}.{name}" if path else name
        typ = (
            _NESTED[f.type] if isinstance(f.type, str) and f.type in _NESTED else f.type
        )

        if is_dataclass(typ):
            kwargs[name] = _build(typ, value, child)
        else:
            kwargs[name] = _coerce(typ, value, child)

    return cls(**kwargs)


def _coerce(typ: Any, value: Any, path: str) -> Any:
    name = typ if isinstance(typ, str) else getattr(typ, "__name__", str(typ))

    if name == "float":
        return float(value)
    if name == "int":
        return value
    if name == "bool":
        return value
    if name == "str":
        return value
    if name == "list":
        args = get_args(typ)
        elem = args[0] if args else str
        return [_coerce(elem, v, f"{path}[{i}]") for i, v in enumerate(value)]

    return value


_NESTED = {
    "DataConfig": DataConfig,
    "SamplerTrainConfig": SamplerTrainConfig,
    "SamplerConfig": SamplerConfig,
    "DiTConfig": DiTConfig,
    "DecoderConfig": DecoderConfig,
    "DecoderTrainConfig": DecoderTrainConfig,
}


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no config at {path}; pass --config or create one at the project root"
        )

    raw = yaml.safe_load(path.read_text()) or {}
    return _build(Config, raw, "")
