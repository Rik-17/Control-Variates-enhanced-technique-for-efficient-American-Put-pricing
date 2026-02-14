# Option Pricing with Control Variate Technique Beyond Monte Carlo Simulation

This project implements the methodology proposed in:

> C. Chiu, T. Dai, Y. Lyuu, L. Liu, Y. Chen  
> *Option pricing with the control variate technique beyond Monte Carlo simulation*  
> The North American Journal of Economics and Finance, Volume 62, 2022, Article 101772  
> DOI: https://doi.org/10.1016/j.najef.2022.101772

The paper extends the control variate technique beyond Monte Carlo simulation and applies it to binomial tree models to improve option pricing accuracy and convergence speed.


## Project Overview

This repository provides Python implementations of control variate–enhanced binomial tree pricing methods as described in the paper.

Two binomial tree models are implemented:

- Cox–Ross–Rubinstein (CRR) model  
- Equal Probabilities (EP) model  

The project also includes classical pricing formulas for benchmarking and comparison purposes.

## Repository Structure

```text
├── images/
│   ├── comparison_table.png
│   └── complete_algorithm.png
│
├── src/
│   ├── main.ipynb  # main jupyter notebook
│   ├── cv_enhanced_crr.py  # algorithm implementation for the CRR binomial tree model
│   └── cv_enhanced_equal_probabilities.py  # algorithm implementation for the Equal Probabilities binomial tree model
│
├── utils/
│   └── standard_pricing_formulas.py  # Various classical pricing formulas
│
└── README.md
```
