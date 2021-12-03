import torch
import pytorch_lightning as pl
import numpy as np

from deepmap import DataModule, Encoder, choose_representation


def autoencode(adata,
               y_label='Metadata_Treatment_Enc',
               use_rep=None,
               n_pcs=50,
               n_epochs=500,
               enc_dims=2,
               lr=1e-3,
               num_workers=0,
               copy=False):
    """Dimensionality reduction with an autoencoder.

    Args:
        adata (anndata.AnnData): Multidimensional morphological data.
        y_label (str): Variable name for labels in .obs.
        use_rep (bool): Make representation of data 3d
        n_pcs (int): Number of PCs to use if use_rep is "X_pca"
        n_epochs (int): Number of Epochs.
        enc_dims (int): Number of dimensions for encoded representation.
        lr (float): Learning rate.
        num_workers (int): Workers used for dataloader.
        copy (bool): Copy anndata object.

    Returns:
        (anndata.AnnData): adata.obsm['X_enc']
    """
    if copy:
        adata = adata.copy()
    # get representation of data
    if use_rep is None:
        use_rep = 'X'
    X = choose_representation(adata,
                              rep=use_rep,
                              n_pcs=n_pcs)

    in_shape = X.shape[-1]
    assert y_label in adata.obs.columns, f"y_label not in .obs: {y_label}"
    y = adata.obs[y_label].to_numpy().flatten()
    n_classes = len(np.unique(y))

    data = DataModule(X_data=X, y_data=y, num_workers=num_workers)
    train_loader = data.train_dataloader()
    valid_loader = data.val_dataloader()
    model = Encoder(in_shape=in_shape, enc_shape=enc_dims,
                        n_classes=n_classes, lr=lr).double()
    trainer = pl.Trainer(max_epochs=n_epochs)
    trainer.fit(model, train_loader, valid_loader)

    with torch.no_grad():
        encoded = model.encode(torch.from_numpy(X).double())
        enc = encoded.cpu().detach().numpy()

    adata.obsm['X_enc'] = enc

    return adata