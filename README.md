# Option Pricing Engine

A modular, extensible option pricing framework in Python supporting **Monte Carlo simulation**, **CRR binomial trees**, and **Black-Scholes closed-form** pricing — all through a unified interface.

---

## Overview

This project implements a clean object-oriented architecture for pricing a wide range of derivatives under Geometric Brownian Motion. The design separates three concerns:

| Module | Responsibility |
|--------|---------------|
| `Option.py` | Payoff definitions, barrier conditions, control variates |
| `process.py` | Stochastic process simulation (`GBM` and future models) |
| `Utilities.py` | Pricing engines: `mc_price`, `crr_price`, `bls_price`, `mc_price_american_lsm` |

Any `Option` can be passed to any compatible pricer — the interface is consistent across all methods.

---

## Supported Instruments

### European Options
| Option | MC | CRR | Black-Scholes |
|--------|:--:|:---:|:-------------:|
| Vanilla Call / Put | ✅ | ✅ | ✅ |
| Digital (Cash-or-Nothing) Call | ✅ | ✅ | ✅ |
| Asset-or-Nothing Call | ✅ | ✅ | ✅ |

### Barrier Options *(discrete monitoring)*
| Option | MC | CRR |
|--------|:--:|:---:|
| Down-and-Out Call / Put | ✅ | ✅ |
| Up-and-Out Call | ✅ | ✅ |
| Double Knock-Out Call | ✅ | ✅ |

### Path-Dependent Options *(MC only)*
| Option | MC |
|--------|:--:|
| Fixed-Strike Lookback Call (minimum) | ✅ |
| Arithmetic Floating-Strike Asian Call | ✅ |

### American Options
| Option | CRR | LSM (MC) |
|--------|:---:|:--------:|
| American Call / Put | ✅ | ✅ |

---

## Variance Reduction

The Monte Carlo engine supports two variance reduction techniques, applied automatically when available:

- **Antithetic variates** — mirrors each random path (`Z` and `−Z`) to reduce variance with no additional simulation cost
- **Control variates** — uses the forward price $S_0 e^{rT}$ as a control with optimal $\beta$ estimated from sample covariance:

$$\hat{V} = \frac{1}{N}\sum_{i=1}^N \left[ h(S^{(i)}) - \hat{\beta}\left(S_T^{(i)} - S_0 e^{rT}\right) \right] e^{-rT}$$

Both are enabled by default and can be disabled via function arguments.

---

## American Option Pricing: Longstaff–Schwartz (LSM)

American options are priced via the LSM algorithm (Longstaff & Schwartz, 2001):

1. Simulate $N$ risk-neutral GBM paths forward
2. At maturity, set $V = \text{payoff}(S_T)$
3. Backward induction: for each step $t$, regress discounted future cashflows on $\{1, S_t, S_t^2\}$ for in-the-money paths
4. Exercise if immediate payoff exceeds the estimated continuation value
5. Average and discount to $t=0$

---

## Project Structure

```
.
├── Option.py                      # Option contracts and payoff definitions
├── process.py                     # Stochastic process base class + GBM
├── Utilities.py                   # Pricing engines (MC, CRR, BLS, LSM)
└── Main_testing_refined.ipynb     # Full demo notebook with theory and results
```

---

## Quick Start

```python
from Option import Option
from process import GBM
from Utilities import mc_price, crr_price, bls_price

# Market parameters
process = GBM(S0=197.2, r=0.05, sigma=0.1566, T=1)
K = 200

# European call — all three pricers
opt = Option.eu_call(K)
bls_p          = bls_price(opt, S0=197.2, r=0.05, T=1, sigma=0.1566)
crr_p          = crr_price(opt, r=0.05, T=1, M=1000, sigma=0.1566, S0=197.2)
mc_p, mc_ci    = mc_price(opt, process, Nsim=100_000)

print(f"Black-Scholes : {bls_p:.4f}")
print(f"CRR           : {crr_p:.4f}")
print(f"Monte Carlo   : {mc_p:.4f}  95% CI: {mc_ci}")

# Down-and-out barrier call
opt_dao = Option.eu_down_and_out_call(K, B=180)
mc_dao, dao_ci = mc_price(opt_dao, process, Nsim=100_000, M=52)

# American put via LSM
from Utilities import mc_price_american_lsm
opt_am = Option.american_put(K)
lsm_p, lsm_ci = mc_price_american_lsm(opt_am, process, Nsim=100_000, M=50)
```

---

## Adding a New Process

Subclass `Process` and implement `simulate()` and `forward()`:

```python
from process import Process
import numpy as np

class MyProcess(Process):
    def __init__(self, S0, r, sigma, T, my_param):
        self.S0, self.r, self.sigma, self.T = S0, r, sigma, T
        self.my_param = my_param

    def simulate(self, Nsim, M, antithetic=True):
        # return np.ndarray of shape (Nsim, M+1)
        ...

    def forward(self):
        return self.S0 * np.exp(self.r * self.T)
```

The new process plugs directly into `mc_price` and `mc_price_american_lsm` with no other changes. Planned extensions include Merton jump-diffusion, Variance Gamma, Normal Inverse Gaussian, and Heston stochastic volatility.

---

## Adding a New Option Type

Subclass or extend `Option` using the class method pattern:

```python
@classmethod
def my_custom_option(cls, K, my_param):
    return cls(
        K,
        payoff_fn    = lambda S, K: ...,   # terminal payoff
        condition_fn = lambda S: ...,       # optional barrier condition (MC)
        requires_path = True,               # set True for path-dependent payoffs
    )
```

---

## Requirements

```
numpy
scipy
matplotlib
```

Install with:

```bash
pip install numpy scipy matplotlib
```

---

## Notebook

`Main_testing_refined.ipynb` walks through all 13 pricing experiments with:
- Mathematical derivations in LaTeX
- Convergence analysis (CRR → Black-Scholes at $O(1/M)$)
- Put-call parity verification
- Early exercise premium analysis
- Barrier discount intuition
- LSM step-by-step walkthrough

---

## Author

**Amirreza Khajouei**  
MSc Mathematical Engineering (Quantitative Finance), Politecnico di Milano  
[GitHub](https://github.com/amirKhaju) · amirreza.khajouei@mail.polimi.it
