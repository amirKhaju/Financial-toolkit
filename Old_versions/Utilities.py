# importing libraries:

import math

from scipy.stats import norm

from Option import Option


def bls_price(option, S0, r, T, sigma):
    """
    Price an option using the closed-form Black-Scholes formula.

    Supports: European call/put, digital call, asset-or-nothing.
    Does NOT support: barrier options, American options (raises NotImplementedError).

    Parameters
    ----------
    option : Option  - option object (see Option class)
    S0     : float   - current underlying price
    r      : float   - continuous risk-free rate
    T      : float   - time to expiry (in years)
    sigma  : float   - annualised volatility

    Returns
    -------
    float : closed-form Black-Scholes price
    """

    if option.crr_condition_fn is not None or option.condition_fn is not None:
        raise NotImplementedError("Black-Scholes does not support barrier options.")
    if option.exercise_fn is not None:
        raise NotImplementedError("Black-Scholes does not support American options.")

    K = option.K

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    disc = np.exp(-r * T)  # discount factor

    # Identify the payoff type by comparing to the known static lambdas on Option
    payoff = option.payoff_fn

    if payoff is Option._call:
        return S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)

    elif payoff is Option._put:
        # Put-call parity: P = C - S0 + K * e^{-rT}
        call = S0 * norm.cdf(d1) - K * disc * norm.cdf(d2)
        return call - S0 + K * disc

    elif payoff is Option._digital_call:
        # Pays $1 if S_T > K: price = e^{-rT} * N(d2)
        return disc * norm.cdf(d2)

    elif payoff is Option._asset_or_nothing:
        # Pays S_T if S_T > K: price = S0 * N(d1)
        return S0 * norm.cdf(d1)

    else:
        raise NotImplementedError(f"No closed-form Black-Scholes formula for this payoff type.")

def crr_price(option, r, T, M, sigma, S0, monitoring_steps=None):

    K = option.K
    dt = T / M

    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u

    if not (d < math.exp(r * dt) < u):
        raise ValueError("Arbitrage detected")

    pi = (math.exp(r * dt) - d) / (u - d)
    pi_d = 1 - pi

    discount = math.exp(-r * dt)
    disc_pi   = discount * pi
    disc_pi_d = discount * pi_d

    # --- Monitoring schedule (FIXED) ---
    if monitoring_steps is None:
        monitor_set = set(range(M + 1))
    else:
        monitor_indices = np.round(np.linspace(0, M, monitoring_steps + 1)).astype(int)
        monitor_set = set(monitor_indices)

    # --- Stock ladder ---
    ud = u / d
    ud_powers = ud ** np.arange(M + 1)

    # --- Terminal ---
    S = S0 * (d ** M) * ud_powers
    values = option.payoff_fn(S, K)

    if (option.crr_condition_fn is not None) and (M in monitor_set):
        values = values * option.crr_condition_fn(S, M)

    # --- Backward ---
    for j in range(M - 1, -1, -1):

        values = disc_pi * values[1:] + disc_pi_d * values[:-1]

        St = S0 * (d ** j) * ud_powers[:j+1]

        # Barrier / condition
        if (option.crr_condition_fn is not None) and (j in monitor_set):
            values = values * option.crr_condition_fn(St, j)

        # Exercise (Bermudan-style)
        if (option.exercise_fn is not None) and (j in monitor_set):
            IV = option.exercise_fn(St, K)
            np.maximum(values, IV, out=values)

    return values[0]

"""
 Monte Carlo option pricer.
 
mc_price takes an Option and a Process and knows nothing about
GBM internals, antithetic construction, or closed-form formulas.
All of that is encapsulated in Option and Process respectively.
"""
import numpy as np

def mc_price(option, process, Nsim, M=1, antithetic=True):

    if option.requires_path and M == 1:
        raise ValueError("This option requires path simulation. Set M > 1.")

    S = process.simulate(Nsim, M, antithetic)   # (Nsim, M+1)

    # --- Payoff ---
    payoff = option.payoff_fn(S, option.K)
    if option.condition_fn is not None:
        payoff = payoff * option.condition_fn(S)

    df = np.exp(-process.r * process.T)

    # --- Control variate ---
    if option.control_fn is not None:
        c_payoff = option.control_fn(S, option.K)          # shape (Nsim,)
        c_price  = option.control_price_fn(                 # scalar, analytical E^Q[C]
            process.S0, process.r, process.sigma, process.T, M
        )

        # Optimal beta using undiscounted quantities (discount cancels in the ratio)
        cov_matrix = np.cov(payoff, c_payoff, ddof=1)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]

        # ✅ Correct formula: adjust for deviation of control FROM its known mean
        adjusted_payoff = payoff - beta * (c_payoff - c_price)
        disc_payoff = df * adjusted_payoff

    else:
        disc_payoff = df * payoff

    # --- Estimation ---
    price     = np.mean(disc_payoff)
    sigma_hat = np.std(disc_payoff, ddof=1)
    ci = (
        price - 1.96 * sigma_hat / np.sqrt(Nsim),
        price + 1.96 * sigma_hat / np.sqrt(Nsim),
    )

    return price, ci

def mc_price_american_lsm(option, process, Nsim, M, antithetic=True):
    """
    Price an American option using Longstaff–Schwartz (LSM).
    """

    # =========================
    # Parameters
    # =========================
    S = process.simulate(Nsim, M, antithetic)   # (Nsim, M+1)
    r = process.r
    T = process.T
    dt = T / M
    df = np.exp(-r * dt)

    # =========================
    # Initial payoff at maturity
    # =========================
    V = option.exercise_fn(S[:, -1], option.K)   # (Nsim,)

    # =========================
    # Backward induction
    # =========================
    for t in range(M - 1, 0, -1):

        St = S[:, t]

        # Immediate exercise value
        exercise_val = option.exercise_fn(St, option.K)

        # In-the-money paths
        itm = exercise_val > 0

        # If no ITM paths → just discount
        if not np.any(itm):
            V *= df
            continue

        # =========================
        # Regression (continuation value)
        # =========================

        # Discount future cashflows
        Y = V * df

        # Basis functions (polynomial)
        X = St[itm]
        A = np.vstack([np.ones_like(X), X, X**2]).T

        # Fit continuation value
        beta = np.linalg.lstsq(A, Y[itm], rcond=None)[0]

        continuation = A @ beta

        # =========================
        # Exercise decision
        # =========================
        exercise_now = exercise_val[itm] > continuation

        # Update values
        V[itm] = np.where(
            exercise_now,
            exercise_val[itm],
            Y[itm]   # continue → use discounted future value
        )

        # Discount non-ITM paths
        V[~itm] *= df

    # =========================
    # Final discount to time 0
    # =========================
    price = np.mean(V * df)

    sigma_hat = np.std(V * df, ddof=1)

    ci = (
        price - 1.96 * sigma_hat / np.sqrt(Nsim),
        price + 1.96 * sigma_hat / np.sqrt(Nsim)
    )

    return price, ci