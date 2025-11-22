from torch.utils.data import Dataset
import torch
from typing import List, Tuple, Callable

class TextPairDataset(Dataset):
    """Simple (text1, text2, label) dataset for contrastive training.

    Each item:
        - text1 (str)
        - text2 (str)
        - label (float: 1.0 for similar, 0.0 for dissimilar)
    """

    def __init__(self, pairs: List[Tuple[str, str, float]]):
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


def make_collate_fn(tokenizer, max_length: int = 256) -> Callable:
    """Return a collate function that tokenizes text pairs."""

    def collate(batch):
        texts1, texts2, labels = zip(*batch)
        enc1 = tokenizer(
            list(texts1),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc2 = tokenizer(
            list(texts2),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return enc1, enc2, labels_tensor

    return collate
