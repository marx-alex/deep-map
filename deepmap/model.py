import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional


class ClusterDistance(nn.Module):
    def __init__(
        self,
        n_classes: int,
        enc_shape: int,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """

        :param n_classes: number of clusters
        :param enc_shape: embedding dimension of feature vectors
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super().__init__()
        self.enc_shape = enc_shape
        self.n_classes = n_classes
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.n_classes, self.enc_shape, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = nn.Parameter(initial_cluster_centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: FloatTensor of [batch size, embedding dimension]
        :param y: FloatTensor of [batch size,]
        :return: FloatTensor [batch size, number of clusters]
        """

        return torch.cdist(x, self.cluster_centers)


class CombinedEncoder(pl.LightningModule):
    """Encoding with Clustering.

    Args:
    in_shape (int): input shape
    enc_shape (int): desired encoded shape
    """

    def __init__(self, in_shape, enc_shape, n_classes, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.enc_shape = enc_shape
        self.n_classes = n_classes

        self.encode = nn.Sequential(
            nn.Linear(in_shape, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(True),
            nn.Linear(16, self.enc_shape),
        )

        self.cluster = nn.Sequential(
            ClusterDistance(self.n_classes, self.enc_shape),
            nn.Tanhshrink(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z = self.encode(x)
        out = self.cluster(z)

        return z, out

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _prepare_batch(self, batch):
        x, y = batch
        return x.view(x.size(0), -1), y

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = self._prepare_batch(batch)
        z, out = self(x)
        loss = F.cross_entropy(out, y)

        self.log(f"{stage}_loss", loss, on_step=True)
        return loss