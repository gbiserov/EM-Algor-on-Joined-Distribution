# EM Algorithm 

A Python /Jupyter notebook/ project exploring the Expectation–Maximization (EM) algorithm.

## Contents

**1. Multinomial model.** A five-category model with cell probabilities
(1/2, θ/4, (1−θ)/4, (1−θ)/4, θ/4). The notebook evaluates the multinomial likelihood for sample counts and derives the MLE of θ.

**2. Gaussian mixture with EM.** It fits a two-component Gaussian mixture to synthetic data (500 points from N(0,1) and 500 from N(5,1)) using the [`mixem`](https://github.com/sseemayer/mixem) library. It then reports the estimated weights, means, standard deviations, and log-likelihood.

The math (likelihoods, E-step, M-step) is explained in markdown cells inside the notebook.

## Requirements

```bash
pip install numpy scipy mixem jupyter
```

## Usage

```bash
jupyter notebook EM_GD.ipynb
```

Run the cells top to bottom. The mixture fit should recover weights ≈ (0.5, 0.5), means ≈ (0, 5), and standard deviations ≈ (1, 1).
