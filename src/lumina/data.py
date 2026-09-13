import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms

from .configs import DataConfig


class Stream(IterableDataset):
    def __init__(self, cfg: DataConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed

        resize = (
            []
            if cfg.preprocessed
            else [
                transforms.Resize(cfg.resolution),
                transforms.CenterCrop(cfg.resolution),
            ]
        )
        self.transform = transforms.Compose([*resize, transforms.ToTensor()])

        self.stream = load_dataset(
            cfg.dataset, split=cfg.split, streaming=not cfg.local
        )

    def __iter__(self):
        stream = self.stream

        if self.cfg.local:
            if self.cfg.shuffle:
                stream = stream.shuffle(seed=self.seed)

            info = get_worker_info()
            if info is not None:
                stream = stream.shard(
                    num_shards=info.num_workers, index=info.id, contiguous=False
                )
        elif self.cfg.shuffle:
            stream = stream.shuffle(
                seed=self.seed, buffer_size=self.cfg.shuffle_buffer
            )

        for example in stream:
            image = example[self.cfg.image_key]
            caption = example[self.cfg.caption_key] or ""

            yield self.transform(image.convert("RGB")), caption


def collate(batch, tokenizer, max_tokens: int):
    pixels = torch.stack([item[0] for item in batch])
    inputs = tokenizer(
        [item[1] for item in batch],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_tokens,
    )

    return pixels, inputs["input_ids"], inputs["attention_mask"]


def collate_pixels(batch):
    return (torch.stack([item[0] for item in batch]),)
