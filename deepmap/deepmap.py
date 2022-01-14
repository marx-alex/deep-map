import torch
import pytorch_lightning as pl
import numpy as np

from deepmap import DataModule, CombinedEncoder


class DeepMap:
    """Dimensionality reduction with an deep encoding.
    """

    def __init__(self,
                 n_epochs=500,
                 enc_dims=2,
                 lr=1e-3,
                 num_workers=0):
        self.n_epochs = n_epochs
        self.enc_dims = enc_dims
        self.lr = lr
        self.num_workers = num_workers

        self.model = CombinedEncoder
        self.data = None

    def fit(self,
            X, y):
        """
        Fit model.

        Args:
            X (np.ndarray): Multidimensional morphological data.
            y (np.ndarray): Labels
        """
        in_shape = X.shape[-1]
        n_classes = len(np.unique(y))

        self.data = DataModule(X_data=X, y_data=y, num_workers=self.num_workers)
        train_loader = self.data.train_dataloader()
        valid_loader = self.data.val_dataloader()
        self.model = CombinedEncoder(in_shape=in_shape, enc_shape=self.enc_dims,
                                     n_classes=n_classes, lr=self.lr).double()
        trainer = pl.Trainer(max_epochs=self.n_epochs)
        trainer.fit(self.model, train_loader, valid_loader)

    def predict(self, return_ix=False):
        """
        Predict embedding on stored test data.
        """
        X_test = self.data.test_dataset.X_data
        ix_test = self.data.ix_test

        with torch.no_grad():
            encoded = self.model.encode(X_test)
            enc = encoded.cpu().detach().numpy()

        if return_ix:
            return enc, ix_test
        return enc

    def save_model(self, path="./"):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)
