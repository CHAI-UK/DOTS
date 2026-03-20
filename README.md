# Diffusion Ordered Temporal Structure (DOTS)

This repository accompanies the paper "Causal Ordering for Structure Learning from Time Series" published in Transactions on Machine Learning Research ([link to the paper](https://openreview.net/forum?id=hWuTzqggSd)).

## One-click Test
Run the following to install necessary packages and execute a demo in a single step:
```
sh setup-run.sh
```

## Manual Approach
Run the following steps in your command line:
```
conda env create -f environment.yml
```

```
conda activate dots
```

```
python demo.py
```

## Interactive Test
Once you have installed necessary packages (see steps above), you can also try an interactive [demo](./demo.ipynb).

## Cite Us

```
@article{
sanchez2025causal,
title={Causal Ordering for Structure Learning from Time Series},
author={Pedro Sanchez and Damian Machlanski and Steven McDonagh and Sotirios A. Tsaftaris},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2025},
url={https://openreview.net/forum?id=hWuTzqggSd}
}
```
