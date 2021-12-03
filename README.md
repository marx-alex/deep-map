# Implementation of DEEP-MAP

This is the Pytorch implementation of DEEP-MAP as described in the original paper by 
Edward Ren:

````
@article {Ren2021.07.31.454574,
	author = {Ren, Edward and Kim, Sungmin and Mohamad, Saad and Huguet, Samuel F. and Shi, Yulin and Cohen, Andrew R. and Piddini, Eugenia and Salas, Rafael Carazo},
	title = {Deep learning-enhanced morphological profiling predicts cell fate dynamics in real-time in hPSCs},
	elocation-id = {2021.07.31.454574},
	year = {2021},
	doi = {10.1101/2021.07.31.454574},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {Predicting how stem cells become patterned and differentiated into target tissues is key for optimising human tissue design. Here, we established DEEP-MAP - for deep learning-enhanced morphological profiling - an approach that integrates single-cell, multi-day, multi-colour microscopy phenomics with deep learning and allows to robustly map and predict cell fate dynamics in real-time without a need for cell state-specific reporters. Using human pluripotent stem cells (hPSCs) engineered to co-express the histone H2B and two-colour FUCCI cell cycle reporters, we used DEEP-MAP to capture hundreds of morphological- and proliferation-associated features for hundreds of thousands of cells and used this information to map and predict spatiotemporally single-cell fate dynamics across germ layer cell fates. We show that DEEP-MAP predicts fate changes as early or earlier than transcription factor-based fate reporters, reveals the timing and existence of intermediate cell fates invisible to fixed-cell technologies, and identifies proliferative properties predictive of cell fate transitions. DEEP-MAP provides a versatile, universal strategy to map tissue evolution and organisation across many developmental and tissue engineering contexts.Competing Interest StatementThe authors have declared no competing interest.},
	URL = {https://www.biorxiv.org/content/early/2021/08/01/2021.07.31.454574},
	eprint = {https://www.biorxiv.org/content/early/2021/08/01/2021.07.31.454574.full.pdf},
	journal = {bioRxiv}
}
````


## Installation

Create a conda environment with all necessary packages and activate
environment.
```
conda env create -f deep-base.yml -n deep-base
conda activate deep-base
```

Install DEEP-MAP.
```
python setup.py install
```