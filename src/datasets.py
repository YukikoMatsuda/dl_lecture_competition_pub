import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint

class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", augment: bool = False) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.augment = augment  # データ拡張を有効にするフラグ
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.augment and self.split == "train":
            X = self.augment_data(X)
        
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

    def augment_data(self, X):
        # ランダムノイズを追加
        noise = torch.randn_like(X) * 0.01
        X = X + noise
        # ランダムシフト
        shift = np.random.randint(-10, 10)
        X = torch.roll(X, shifts=shift, dims=-1)
        return X
