import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from dataset import CatDogDataset


class CatDogDataModule(pl.LightningDataModule):
    """
    Args:
        root: (str or Path): Path to the directory where the dataset is found or downloaded.
        batch_size: (int) default 64
    """

    def __init__(self, root: str = 'data', batch_size: int = 64, **kwargs):
        super().__init__()
        self.test_set = None
        self.val_set = None
        self.train_set = None

        self.batch_size = batch_size
        self.root = root
        self.setup()

    def setup(self, stage=None):
        self.train_set = CatDogDataset(root=self.root, split="train")
        self.val_set = CatDogDataset(root=self.root, split="val")
        self.test_set = CatDogDataset(root=self.root, split="test")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                           num_workers=os.cpu_count(), drop_last=True,
                                           pin_memory=True,
                                           )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                           num_workers=os.cpu_count(), drop_last=False,
                                           pin_memory=True,
                                           )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                           num_workers=os.cpu_count(), drop_last=False,
                                           pin_memory=True,
                                           )
