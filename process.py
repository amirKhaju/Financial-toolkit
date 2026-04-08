"""
process.py — Stochastic process base class and concrete implementations.

Each process encapsulates all market parameters and knows how to simulate
risk-neutral paths. mc_price only ever calls process.simulate(...).

Exports
-------
Process  : abstract base class
GBM      : Geometric Brownian Motion under risk-neutral measure
"""

import numpy as np
from abc import ABC, abstractmethod


class Process(ABC):
    """
    Abstract base class for risk-neutral stochastic processes.

    Subclasses must implement:
      - simulate(Nsim, M, antithetic) → S of shape (Nsim, M+1)
      - forward()                     → E^Q[S_T] = analytical forward price
    """

    @abstractmethod
    def simulate(self, Nsim, M, antithetic=True):
        """
        Simulate risk-neutral price paths.

        Parameters
        ----------
        Nsim       : int  - number of simulated paths
        M          : int  - number of time steps (path has M+1 columns incl. S0)
        antithetic : bool - whether to use antithetic variates

        Returns
        -------
        S : np.ndarray, shape (Nsim, M+1)
            Simulated price paths. S[:, 0] = S0 for all paths.
        """

    @abstractmethod
    def forward(self):
        """
        Analytical risk-neutral expectation E^Q[S_T].
        Used as the control variate price for the forward control.
        """


class GBM(Process):
    """
    Geometric Brownian Motion under the risk-neutral measure.

    dS = r * S * dt + sigma * S * dW

    Parameters
    ----------
    S0    : float - current underlying price
    r     : float - continuous risk-free rate
    sigma : float - annualised volatility
    T     : float - time to expiry (in years)
    """

    def __init__(self, S0, r, sigma, T):
        self.S0    = S0
        self.r     = r
        self.sigma = sigma
        self.T     = T

    def simulate(self, Nsim, M, antithetic=True):
        """
        Simulate GBM paths via Euler discretisation of log-returns.
        If antithetic=True, simulates Nsim/2 paths and mirrors them,
        """
        dt   = self.T / M
        half = Nsim // 2

        Z = np.random.randn(half, M)
        if antithetic:
            Z = np.vstack([Z, -Z])          # shape (Nsim, M)
        else:
            Z = np.random.randn(Nsim, M)

        increments = (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z
        X = np.cumsum(increments, axis=1)                    # cumulative log-returns
        X = np.hstack([np.zeros((Nsim, 1)), X])              # prepend 0 → S[:, 0] = S0

        return self.S0 * np.exp(X)                           # shape (Nsim, M+1)

    def forward(self):
        """E^Q[S_T] = S0 * exp(r * T)"""
        return self.S0 * np.exp(self.r * self.T)