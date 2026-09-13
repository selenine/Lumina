# Lumina

Lumina is a few-step latent diffusion transformer (DiT) trained using the MeanFlow objective, that I built to learn how DiTs work. Unlike standard MeanFlow, I decided to perform denoising in DINO-space as opposed to using a VAE, with conditioning via cross-attention from CLIP embeddings. 

## Usage

Edit `config.yaml`, then 

```sh
uv sync
uv run accelerate launch --config_file accelerate.yaml -m lumina.cli train-decoder # train RAE decoder
uv run accelerate launch --config_file accelerate.yaml -m lumina.cli train-sampler # train the model itself
uv run lumina generate "an oil painting of a lighthouse on a stormy coast"
```

Note that the DINOv3 weights are gated, so run `hf auth login` first.

## Config

Every key in `config.yaml` is optional and falls back to `src/lumina/configs.py`; use `lumina config` to print the full configuration used.

To use a different config, make use of the `--config` kewyord argument.

## Guidance

To use classifier-free guidance, use the `-g` or `--guidance` flag

```sh
uv run lumina generate "a watercolor landscape of mountains at sunset" -g 3.0   # -g 1.0 turns guidance off
```
