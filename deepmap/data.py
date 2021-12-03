import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
import numpy as np


class ClassifierDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class DataModule(pl.LightningDataModule):
    def __init__(self, X_data: np.array, y_data: np.array, batch_size: int = 32,
                 num_workers=0):
        super().__init__()

        X_trainval, X_test, y_trainval, y_test = train_test_split(X_data, y_data,
                                                                  test_size=0.2, stratify=y_data,
                                                                  random_state=0)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval,
                                                          test_size=0.1, stratify=y_trainval,
                                                          random_state=0)
        self.test_dataset = ClassifierDataset(torch.from_numpy(X_test).double(),
                                              torch.from_numpy(y_test).long())
        self.train_dataset = ClassifierDataset(torch.from_numpy(X_train).double(),
                                               torch.from_numpy(y_train).long())
        self.val_dataset = ClassifierDataset(torch.from_numpy(X_val).double(),
                                             torch.from_numpy(y_val).long())
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)