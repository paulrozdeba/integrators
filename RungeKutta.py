"""
A family of Runge-Kutta integrators.
"""

import numpy as np

def RK4(f, x, t1, t2, pf, stim=None):
    """
    Fourth-order, 4-step RK routine.
    Returns the step, i.e. approximation to the integral.
    If x is defined at time t_1, then stim should be an array of
    stimulus values at times t_1, (t_1+t_2)/2, and t_2 (i.e. at t1 and t2, as
    well as at the midpoint).

    Alternatively, stim may be a function pointer.
    """
    tmid = (t1 + t2)/2.0
    dt = t2 - t1

    if stim is None:
        pf_in_1 = pf
        pf_in_mid = pf
        pf_in_2 = pf
    else:
        try:
            # test if stim is a function
            s1 = stim(t1)
            s1, smid, s2 = (stim, stim, stim)
        except TypeError:
            #  otherwise assume stim is an array
            s1, smid, s2 = (stim[0], stim[1], stim[2])
        pf_in_1 = (pf, s1)
        pf_in_mid = (pf, smid)
        pf_in_2 = (pf, s2)

    K1 = f(t1, x, pf_in_1)
    K2 = f(tmid, x + dt*K1/2.0, pf_in_mid)
    K3 = f(tmid, x + dt*K2/2.0, pf_in_mid)
    K4 = f(t2, x + dt*K3, pf_in_2)

    return dt * (K1/2.0 + K2 + K3 + K4/2.0) / 3.0

def RK4_vec(f, x, t, pf, stim=None):
    """
    Vectorized RK4, for evaluating over a whole trajectory.
    If x is defined at time points (t_1, ..., t_n), then stim should be an
    array defined at times (t_1, ..., t_n) as well as at the midpoints,
    PLUS at t_n+1 and (t_n+1 + t_n)/2.  Thus, x has length n, but stim has
    length n + (n-1) + 2 = 2n + 1.

    Alternatively, stim can be a vectorized function of t.
    """
    NT,D = x.shape
    t1, t2 = (t[:-1], t[1:])
    tmid = (t1 + t2)/2.0
    dt = t2 - t1
    dt = np.tile(dt, (D,1)).T

    if stim is None:
        pf_in_1 = pf
        pf_in_mid = pf
        pf_in_2 = pf
    else:
        try:
            s1, smid, s2 = (stim(t1), stim(tmid), stim(t2))
        except TypeError:
            s1, smid, s2 = (stim[:-2:2], stim[1:-1:2], stim[2::2])
        pf_in_1 = (pf, s1)
        pf_in_mid = (pf, smid)
        pf_in_2 = (pf, s2)

    K1 = f(t1, x, pf_in_1)
    K2 = f(tmid, x + dt*K1/2.0, pf_in_mid)
    K3 = f(tmid, x + dt*K2/2.0, pf_in_mid)
    K4 = f(t2, x + dt*K3, pf_in_2)

    return K1/6.0 + K2/3.0 + K3/3.0 + K4/6.0

################################################################################
# RK4 versions which assume constant stim over time window.
################################################################################
def RK4_conststim(f, x, t1, t2, pf, stim=None):
    """
    Fourth-order, 4-step RK routine.
    Returns the step, i.e. approximation to the integral.
    """
    dt = t2 - t1

    if stim is None:
        pf_in = pf
    else:
        try:
            s1 = stim(t1)
            s1 = stim
        except TypeError:
            s1 = stim
        pf_in = (pf, s1)

    K1 = f(t1, x, pf_in)
    K2 = f(t1, x + dt*K1/2.0, pf_in)
    K3 = f(t1, x + dt*K2/2.0, pf_in)
    K4 = f(t1, x + dt*K3, pf_in)

    return dt * (K1/6.0 + K2/3.0 + K3/3.0 + K4/6.0)

def RK4_vec_conststim(f, x, t, pf, stim=None):
    """
    Vectorized RK4, for evaluating over a whole trajectory.
    """
    NT,D = x.shape
    dt = t[1] - t[0]
    dt = np.tile(dt, (D,1)).T

    if stim is None:
        pf_in = pf
    else:
        try:
            s1 = stim(t)
        except TypeError:
            s1 = stim
        pf_in = (pf, s1)

    K1 = f(t, x, pf_in)
    K2 = f(t, x + dt*K1/2.0, pf_in)
    K3 = f(t, x + dt*K2/2.0, pf_in)
    K4 = f(t, x + dt*K3, pf_in)

    return dt * (K1/6.0 + K2/3.0 + K3/3.0 + K4/6.0)

def RK4_best_conststim(f, x, t, pf, stim=None, dt=None):
    """
    Vectorized RK4, for evaluating over a whole trajectory.
    """
    NT,D = x.shape
    if NT > 1:
        dt = t[1] - t[0]
        dt = np.tile(dt, (D,1)).T

    if stim is None:
        pf_in = pf
    else:
        try:
            s1 = stim(t)
        except TypeError:
            s1 = stim
        pf_in = (pf, s1)

    K1 = f(t, x, pf_in)
    K2 = f(t, x + dt*K1/2.0, pf_in)
    K3 = f(t, x + dt*K2/2.0, pf_in)
    K4 = f(t, x + dt*K3, pf_in)

    return dt * (K1/6.0 + K2/3.0 + K3/3.0 + K4/6.0)

################################################################################
# CODE STAGING AREA
# The RK4 method needs the stimulus at the next time point, so it is impossible
# to evaluate RK4 at the end of a time series.
################################################################################
def RK4_vec_all(f, x, t, pf, stim=None):
    """
    Vectorized RK4, for evaluating over a whole trajectory.
    Evaluates at all x INCLUDING the last entry.  Assumes dt is fixed.
    """
    if x.ndim == 1:
        NT = 1
        D = x.shape[0]
    else:
        NT,D = x.shape

    if NT > 1:
        dt = t[1] - t[0]
    dt = dt * np.ones((NT,D))

    if stim is None:
        pf_in_1 = pf
        pf_in_mid = pf
        pf_in_2 = pf

    K1 = np.copy(x)
    K2 = x + dt*f(t,K1,pf)/2.0
    K3 = x + dt*f(t,K2,pf)/2.0
    K4 = x + dt*f(t,K3,pf)

    return f(t,K1,pf)/6.0 + f(t,K2,pf)/3.0 + f(t,K3,pf)/3.0 + f(t,K4,pf)/6.0

def RK4_vec_xt(f, x, t, pf):
    """
    Vectorized RK4, for evaluating over a whole trajectory.
    """
    NT,D = x.shape
    t1,t2 = (t[:-1],t[1:])
    tmid = (t1 + t2)/2.0
    dt = t2 - t1
    dt = np.tile(dt, (D,1)).T

    x1 = x[:-1]
    K1 = f(x, t1, pf)
    K2 = f(x + dt*K2/2.0, tmid, pf)
    K3 = f(x + dt*K3/2.0, tmid, pf)
    K4 = f(x + dt*K3, t2, pf)
    return dt * (K1/6.0 + K2/3.0 + K3/3.0 + K4/6.0)
