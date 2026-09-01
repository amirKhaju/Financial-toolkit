"""
process.py — Stochastic process base class and concrete implementations.

Each process encapsulates all market parameters and knows how to simulate
risk-neutral paths. mc_price only ever calls process.simulate(...).

Exports
-------
Process               : abstract base class
GBM                   : Geometric Brownian Motion under risk-neutral measure
MertonJumpDiffusion    : GBM + compound Poisson jumps, Normal jump sizes
KouJumpDiffusion       : GBM + compound Poisson jumps, asymmetric double-exponential jump sizes
VarianceGamma          : pure-jump Lévy process, Gamma time-changed Brownian motion
NIG                    : pure-jump Lévy process, Inverse-Gaussian time-changed Brownian motion

All jump/Lévy processes are simulated *exactly* (no Euler discretisation error):
compound-Poisson jump counts are exact for any dt, and the Gamma / Inverse-Gaussian
subordinators used for VG / NIG have exact samplers, so path granularity (M) only
matters for path-dependent payoffs (barriers, lookbacks, LSM), not for accuracy of S_T.

Every process also exposes an (optional) `characteristic_function(u, T)`, giving
    phi(u, T) = E^Q[exp(i * u * ln(S_T / S0))]
which is all `fft_price` (Carr-Madan) in Utilities.py needs to price European
vanillas — no simulation required.
"""

import numpy as np
from scipy.stats import invgauss
from abc import ABC, abstractmethod


class Process(ABC):
    """
    Abstract base class for risk-neutral stochastic processes.

    Subclasses must implement:
      - simulate(Nsim, M, antithetic) → S of shape (Nsim, M+1)
      - forward()                     → E^Q[S_T] = analytical forward price

    Subclasses MAY implement:
      - characteristic_function(u, T) → E^Q[exp(i*u*ln(S_T/S0))]
        Needed only for FFT (Carr-Madan) pricing; simulation-based pricers
        (mc_price, mc_price_american_lsm) never call it.
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

    def characteristic_function(self, u, T):
        """
        Characteristic function of X_T = ln(S_T / S0) under the risk-neutral measure:

            phi(u, T) = E^Q[exp(i * u * X_T)]

        Optional — only required to use `fft_price`. Default raises, so processes
        that don't implement it fail loudly rather than silently mispricing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement characteristic_function(); "
            "FFT pricing is unavailable for this process."
        )


# =============================================================================
# Shared helper: exact vectorised simulation of compound-Poisson jump increments
# =============================================================================

def _poisson_jump_increments(Nsim, M, dt, lam, sample_jump_sizes):
    """
    Exactly simulate compound-Poisson jump increments for a (Nsim, M) grid.

    For each cell, the number of jumps N ~ Poisson(lam * dt) is exact for any dt.
    Given N, the jump sizes are drawn iid from `sample_jump_sizes`. This is done
    without a Python loop over paths/steps by adding jump sizes "layer by layer":
    layer k contributes a jump to every cell with N >= k.

    Parameters
    ----------
    Nsim, M            : grid shape
    dt                 : float, time step
    lam                : float, jump intensity (lambda)
    sample_jump_sizes  : callable(n) -> ndarray of n iid jump sizes (log-price scale)

    Returns
    -------
    jumps : np.ndarray, shape (Nsim, M) — total jump contribution to log-returns
    """
    N = np.random.poisson(lam * dt, size=(Nsim, M))
    max_jumps = int(N.max()) if N.size else 0

    jumps = np.zeros((Nsim, M))
    for k in range(1, max_jumps + 1):
        mask = N >= k
        n_active = int(mask.sum())
        if n_active == 0:
            continue
        jumps[mask] += sample_jump_sizes(n_active)
    return jumps


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

    def characteristic_function(self, u, T):
        """
        X_T = ln(S_T/S0) = (r - 0.5*sigma^2)*T + sigma*sqrt(T)*Z,  Z ~ N(0,1)
        phi(u,T) = exp(i*u*(r - 0.5*sigma^2)*T - 0.5*sigma^2*T*u^2)
        """
        u = np.asarray(u, dtype=complex)
        mean = (self.r - 0.5 * self.sigma ** 2) * T
        return np.exp(1j * u * mean - 0.5 * self.sigma ** 2 * T * u ** 2)


# =============================================================================
# Merton jump-diffusion
# =============================================================================

class MertonJumpDiffusion(Process):
    """
    Merton (1976) jump-diffusion under the risk-neutral measure.

    dS/S = (r - lambda*k) dt + sigma dW + d( sum_{i=1}^{N_t} (e^{Y_i} - 1) )

    Jump sizes in log-price:  Y_i ~ Normal(mu_j, sigma_j^2), iid
    Jump arrivals:            N_t ~ Poisson(lambda * t)
    Compensator:              k = E[e^Y] - 1 = exp(mu_j + 0.5*sigma_j^2) - 1

    Parameters
    ----------
    S0, r, sigma, T : as in GBM
    lam             : float - jump intensity (expected jumps per year)
    mu_j, sigma_j   : float - mean and std of the log-jump size
    """

    def __init__(self, S0, r, sigma, T, lam, mu_j, sigma_j):
        self.S0, self.r, self.sigma, self.T = S0, r, sigma, T
        self.lam, self.mu_j, self.sigma_j = lam, mu_j, sigma_j

    def _compensator(self):
        return np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1

    def simulate(self, Nsim, M, antithetic=True):
        dt = self.T / M
        half = Nsim // 2

        if antithetic:
            Z = np.random.randn(half, M)
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.randn(Nsim, M)

        k = self._compensator()
        drift = (self.r - 0.5 * self.sigma ** 2 - self.lam * k) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        jumps = _poisson_jump_increments(
            Nsim, M, dt, self.lam,
            lambda n: np.random.normal(self.mu_j, self.sigma_j, size=n)
        )

        increments = drift + diffusion + jumps
        X = np.cumsum(increments, axis=1)
        X = np.hstack([np.zeros((Nsim, 1)), X])
        return self.S0 * np.exp(X)

    def forward(self):
        return self.S0 * np.exp(self.r * self.T)

    def characteristic_function(self, u, T):
        u = np.asarray(u, dtype=complex)
        k = self._compensator()
        diffusion_part = (
            1j * u * (self.r - 0.5 * self.sigma ** 2 - self.lam * k) * T
            - 0.5 * self.sigma ** 2 * T * u ** 2
        )
        jump_part = self.lam * T * (
            np.exp(1j * u * self.mu_j - 0.5 * self.sigma_j ** 2 * u ** 2) - 1
        )
        return np.exp(diffusion_part + jump_part)


# =============================================================================
# Kou double-exponential jump-diffusion
# =============================================================================

class KouJumpDiffusion(Process):
    """
    Kou (2002) double-exponential jump-diffusion under the risk-neutral measure.

    dS/S = (r - lambda*k) dt + sigma dW + d( sum_{i=1}^{N_t} (e^{Y_i} - 1) )

    Jump sizes in log-price are asymmetric double-exponential:
        Y_i =  +Exp(eta1)  with prob. p     (upward jump)
        Y_i =  -Exp(eta2)  with prob. 1-p   (downward jump)
    Compensator:
        k = E[e^Y] - 1 = p*eta1/(eta1-1) + (1-p)*eta2/(eta2+1) - 1
        (requires eta1 > 1 for E[e^Y] to be finite)

    Parameters
    ----------
    S0, r, sigma, T : as in GBM
    lam             : float - jump intensity
    p               : float in (0,1) - probability of an upward jump
    eta1            : float (> 1) - rate of the upward exponential leg
    eta2            : float (> 0) - rate of the downward exponential leg
    """

    def __init__(self, S0, r, sigma, T, lam, p, eta1, eta2):
        if eta1 <= 1:
            raise ValueError("eta1 must be > 1 for E[e^Y] (and the risk-neutral "
                              "compensator) to be finite.")
        self.S0, self.r, self.sigma, self.T = S0, r, sigma, T
        self.lam, self.p, self.eta1, self.eta2 = lam, p, eta1, eta2

    def _compensator(self):
        p, eta1, eta2 = self.p, self.eta1, self.eta2
        return p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

    @staticmethod
    def _sample_jumps(n, p, eta1, eta2):
        up = np.random.rand(n) < p
        sizes = np.empty(n)
        n_up = int(up.sum())
        n_down = n - n_up
        sizes[up] = np.random.exponential(1.0 / eta1, size=n_up)
        sizes[~up] = -np.random.exponential(1.0 / eta2, size=n_down)
        return sizes

    def simulate(self, Nsim, M, antithetic=True):
        dt = self.T / M
        half = Nsim // 2

        if antithetic:
            Z = np.random.randn(half, M)
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.randn(Nsim, M)

        k = self._compensator()
        drift = (self.r - 0.5 * self.sigma ** 2 - self.lam * k) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z

        jumps = _poisson_jump_increments(
            Nsim, M, dt, self.lam,
            lambda n: self._sample_jumps(n, self.p, self.eta1, self.eta2)
        )

        increments = drift + diffusion + jumps
        X = np.cumsum(increments, axis=1)
        X = np.hstack([np.zeros((Nsim, 1)), X])
        return self.S0 * np.exp(X)

    def forward(self):
        return self.S0 * np.exp(self.r * self.T)

    def characteristic_function(self, u, T):
        u = np.asarray(u, dtype=complex)
        p, eta1, eta2 = self.p, self.eta1, self.eta2
        k = self._compensator()

        diffusion_part = (
            1j * u * (self.r - 0.5 * self.sigma ** 2 - self.lam * k) * T
            - 0.5 * self.sigma ** 2 * T * u ** 2
        )
        phi_jump = p * eta1 / (eta1 - 1j * u) + (1 - p) * eta2 / (eta2 + 1j * u)
        jump_part = self.lam * T * (phi_jump - 1)
        return np.exp(diffusion_part + jump_part)


# =============================================================================
# Variance Gamma (pure-jump, infinite activity)
# =============================================================================

class VarianceGamma(Process):
    """
    Variance Gamma process (Madan, Carr & Chang, 1998) under the risk-neutral measure.

    Constructed as Brownian motion with drift, time-changed by a Gamma subordinator:
        X_t = theta * G_t + sigma * W(G_t),      G_t ~ Gamma(shape=t/nu, scale=nu)
    Risk-neutral drift correction (so that E^Q[S_T] = S0*e^{rT}):
        omega = (1/nu) * ln(1 - theta*nu - 0.5*sigma^2*nu)     [requires the log's argument > 0]
        ln(S_t/S0) = (r + omega) * t + X_t

    Parameters
    ----------
    S0, r, T : as in GBM
    sigma    : float - volatility of the Brownian motion being subordinated
    nu       : float (> 0) - variance rate of the Gamma time change (controls kurtosis)
    theta    : float - drift of the Brownian motion being subordinated (controls skew)
    """

    def __init__(self, S0, r, T, sigma, nu, theta):
        arg = 1 - theta * nu - 0.5 * sigma ** 2 * nu
        if arg <= 0:
            raise ValueError("1 - theta*nu - 0.5*sigma^2*nu must be > 0 for the "
                              "risk-neutral drift correction to be real-valued.")
        self.S0, self.r, self.T = S0, r, T
        self.sigma, self.nu, self.theta = sigma, nu, theta

    def _omega(self):
        return (1.0 / self.nu) * np.log(1 - self.theta * self.nu - 0.5 * self.sigma ** 2 * self.nu)

    def simulate(self, Nsim, M, antithetic=True):
        dt = self.T / M
        half = Nsim // 2

        if antithetic:
            Z = np.random.randn(half, M)
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.randn(Nsim, M)

        # Gamma subordinator increments: iid Gamma(dt/nu, nu) across steps (independent increments)
        G = np.random.gamma(shape=dt / self.nu, scale=self.nu, size=(Nsim, M))

        omega = self._omega()
        increments = (self.r + omega) * dt + self.theta * G + self.sigma * np.sqrt(G) * Z
        X = np.cumsum(increments, axis=1)
        X = np.hstack([np.zeros((Nsim, 1)), X])
        return self.S0 * np.exp(X)

    def forward(self):
        return self.S0 * np.exp(self.r * self.T)

    def characteristic_function(self, u, T):
        # phi_VG(u;t) = (1 - i*u*theta*nu + 0.5*sigma^2*nu*u^2)^(-t/nu)
        # (derived as the Gamma-subordinated MGF: E_G[e^{iu*theta*G - 0.5*sigma^2*u^2*G}])
        u = np.asarray(u, dtype=complex)
        omega = self._omega()
        drift_part = np.exp(1j * u * (self.r + omega) * T)
        vg_part = (1 - 1j * u * self.theta * self.nu + 0.5 * self.sigma ** 2 * self.nu * u ** 2) ** (-T / self.nu)
        return drift_part * vg_part


# =============================================================================
# Normal Inverse Gaussian (pure-jump, infinite activity)
# =============================================================================

class NIG(Process):
    """
    Normal Inverse Gaussian process (Barndorff-Nielsen, 1997) under the
    risk-neutral measure.

    Constructed as Brownian motion time-changed by an Inverse-Gaussian subordinator:
        X_t = beta * I_t + sqrt(I_t) * Z,     I_t ~ IG(mean = delta*t/gamma, shape = (delta*t)^2)
        gamma = sqrt(alpha^2 - beta^2)
    Risk-neutral drift correction:
        omega = delta * ( sqrt(alpha^2 - (beta+1)^2) - gamma )
        ln(S_t/S0) = (r + omega) * t + X_t

    Parameters
    ----------
    S0, r, T   : as in GBM
    alpha      : float (> 0) - steepness (tail heaviness); alpha > |beta|
    beta       : float - asymmetry / skewness parameter
    delta      : float (> 0) - scale parameter
    """

    def __init__(self, S0, r, T, alpha, beta, delta):
        if not (alpha > abs(beta)):
            raise ValueError("NIG requires alpha > |beta|.")
        self.S0, self.r, self.T = S0, r, T
        self.alpha, self.beta, self.delta = alpha, beta, delta
        # sigma kept as None: no direct "volatility" parameter for NIG, but this
        # attribute is expected by the generic forward-control-variate interface
        # in mc_price (Option._cv_forward_price accepts, but ignores, sigma).
        self.sigma = None

    def _gamma(self):
        return np.sqrt(self.alpha ** 2 - self.beta ** 2)

    def _omega(self):
        g = self._gamma()
        return self.delta * (np.sqrt(self.alpha ** 2 - (self.beta + 1) ** 2) - g)

    def simulate(self, Nsim, M, antithetic=True):
        dt = self.T / M
        half = Nsim // 2

        if antithetic:
            Z = np.random.randn(half, M)
            Z = np.vstack([Z, -Z])
        else:
            Z = np.random.randn(Nsim, M)

        gamma = self._gamma()
        ig_mean = self.delta * dt / gamma
        ig_shape = (self.delta * dt) ** 2
        # scipy.stats.invgauss(mu, scale=lam) samples InverseGaussian(mean=mu*lam, shape=lam)
        mu_scipy = ig_mean / ig_shape
        I = invgauss.rvs(mu_scipy, scale=ig_shape, size=(Nsim, M))

        omega = self._omega()
        increments = (self.r + omega) * dt + self.beta * I + np.sqrt(I) * Z
        X = np.cumsum(increments, axis=1)
        X = np.hstack([np.zeros((Nsim, 1)), X])
        return self.S0 * np.exp(X)

    def forward(self):
        return self.S0 * np.exp(self.r * self.T)

    def characteristic_function(self, u, T):
        u = np.asarray(u, dtype=complex)
        gamma = self._gamma()
        omega = self._omega()
        drift_part = np.exp(1j * u * (self.r + omega) * T)
        nig_part = np.exp(T * self.delta * (gamma - np.sqrt(self.alpha ** 2 - (self.beta + 1j * u) ** 2)))
        return drift_part * nig_part