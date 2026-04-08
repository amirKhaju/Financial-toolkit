from functools import partial
import numpy as np


class Option:

    # =========================
    # Helpers
    # =========================

    @staticmethod
    def _terminal(S):
        """Return terminal price(s), works for both MC (2D) and CRR (1D)."""
        return S[:, -1] if S.ndim == 2 else S


    # =========================
    # Payoffs
    # =========================

    _call = staticmethod(lambda S, K: np.maximum(Option._terminal(S) - K, 0))

    _put  = staticmethod(lambda S, K: np.maximum(K - Option._terminal(S), 0))

    _digital_call = staticmethod(
        lambda S, K: (Option._terminal(S) > K).astype(float)
    )

    _asset_or_nothing = staticmethod(
        lambda S, K: Option._terminal(S) * (Option._terminal(S) > K).astype(float)
    )

    # Path-dependent payoff (MC only)
    _lb_call_min = staticmethod(
        lambda S, K: np.maximum(np.min(S, axis=1) - K, 0)
    )

    # Asian payoff (MC only)
    _arithmetic_floating_strike_Asian_call = staticmethod(
        lambda S, K=None: np.maximum(Option._terminal(S) - np.average(S[:, 1:], axis=1), 0)
    )


    # =========================
    # Condition functions — MC
    # =========================

    _mc_down_and_out = staticmethod(
        lambda S, B: (S.min(axis=1) > B).astype(float)
    )

    _mc_up_and_out = staticmethod(
        lambda S, B: (S.max(axis=1) < B).astype(float)
    )

    _mc_double_knock_out = staticmethod(
        lambda S, B_low, B_high: (
            (S.min(axis=1) > B_low) & (S.max(axis=1) < B_high)
        ).astype(float)
    )


    # =========================
    # Condition functions — CRR
    # =========================

    _crr_down_and_out = staticmethod(
        lambda S, j, B: (S > B).astype(float)
    )

    _crr_up_and_out = staticmethod(
        lambda S, j, B: (S < B).astype(float)
    )

    _crr_double_knock_out = staticmethod(
        lambda S, j, B_low, B_high: ((S > B_low) & (S < B_high)).astype(float)
    )


    # =========================
    # Control variates
    # =========================

    # Control payoff: F = S_T (the underlying terminal price)
    # Applicable to any option whose payoff depends on S_T.
    _cv_forward_payoff = staticmethod(
        lambda S, K: S[:, -1]
    )

    # Control price: E^Q[S_T] = S0 * exp(r * T)
    # This is the only analytical expectation needed; no closed-form import required.
    _cv_forward_price = staticmethod(
        lambda S0, r, sigma, T, M: S0 * np.exp(r * T)
    )


    # =========================
    # Constructor
    # =========================

    def __init__(self, K=None, payoff_fn=None, condition_fn=None,
                 crr_condition_fn=None, exercise_fn=None,
                 requires_path=False, control_fn=None, control_price_fn=None):

        self.K                = K
        self.payoff_fn        = payoff_fn
        self.condition_fn     = condition_fn
        self.crr_condition_fn = crr_condition_fn
        self.exercise_fn      = exercise_fn
        self.requires_path    = requires_path
        self.control_fn       = control_fn
        self.control_price_fn = control_price_fn


    # =========================
    # European options
    # =========================

    @classmethod
    def eu_call(cls, K):
        return cls(
            K,
            payoff_fn        = cls._call,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )

    @classmethod
    def eu_put(cls, K):
        return cls(
            K,
            payoff_fn        = cls._put,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )

    @classmethod
    def eu_digital_call(cls, K):
        return cls(
            K,
            payoff_fn        = cls._digital_call,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )

    @classmethod
    def eu_asset_or_nothing(cls, K):
        return cls(
            K,
            payoff_fn        = cls._asset_or_nothing,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )


    # =========================
    # Barrier options
    # =========================

    @classmethod
    def eu_down_and_out_call(cls, K, B):
        return cls(
            K,
            payoff_fn        = cls._call,
            condition_fn     = partial(cls._mc_down_and_out,  B=B),
            crr_condition_fn = partial(cls._crr_down_and_out, B=B),
            requires_path    = True,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )

    @classmethod
    def eu_up_and_out_call(cls, K, B):
        return cls(
            K,
            payoff_fn        = cls._call,
            condition_fn     = partial(cls._mc_up_and_out,  B=B),
            crr_condition_fn = partial(cls._crr_up_and_out, B=B),
            requires_path    = True,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )

    @classmethod
    def eu_down_and_out_put(cls, K, B):
        return cls(
            K,
            payoff_fn        = cls._put,
            condition_fn     = partial(cls._mc_down_and_out,  B=B),
            crr_condition_fn = partial(cls._crr_down_and_out, B=B),
            requires_path    = True,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )

    @classmethod
    def eu_double_knock_out_call(cls, K, B_low, B_high):
        return cls(
            K,
            payoff_fn        = cls._call,
            condition_fn     = partial(cls._mc_double_knock_out,  B_low=B_low, B_high=B_high),
            crr_condition_fn = partial(cls._crr_double_knock_out, B_low=B_low, B_high=B_high),
            requires_path    = True,
            control_fn       = cls._cv_forward_payoff,
            control_price_fn = cls._cv_forward_price,
        )


    # =========================
    # Lookback (MC only)
    # =========================

    @classmethod
    def fixed_strike_lookback_call_min(cls, K):
        return cls(
            K,
            payoff_fn     = cls._lb_call_min,
            requires_path = True,
            # No control variate: no standard closed-form for lookback options
        )


    # =========================
    # Asian (MC only)
    # =========================

    @classmethod
    def arithmetic_floating_strike_asian_call(cls):
        return cls(
            payoff_fn     = cls._arithmetic_floating_strike_Asian_call,
            requires_path = True,
            # No control variate: floating strike makes geometric Asian CV inapplicable
        )


    # =========================
    # American options (CRR)
    # =========================

    @classmethod
    def american_call(cls, K):
        return cls(K, payoff_fn=cls._call, exercise_fn=cls._call)

    @classmethod
    def american_put(cls, K):
        return cls(K, payoff_fn=cls._put, exercise_fn=cls._put)