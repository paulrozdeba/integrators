"""
Euler-Maruyama method for SODE integration.
"""

import numpy as np

def euler_maruyama(f, g, x, t1, t2, pf, pg, sidx, fstim=None):
    """
    Stochastic Euler-Maruyama method, strong order 1/2.
    """
    dt = t2 - t1
    D = len(x)
    Nrand = len(sidx)

    if fstim is None:
        pf_in_1 = pf
    else:
        try:
            s1 = fstim(t1)
        except TypeError:
            s1 = fstim
        pf_in_1 = (pf, s1)

    xi = np.sqrt(dt) * np.random.randn(Nrand)
    dx = f(t1, x, pf_in_1) * dt
    dx[sidx] += g(t1, x, pg) * xi

    return dx

def euler_maruyama_xt(f, g, x, t1, t2, pf, pg, sidx, fstim=None):
    """
    Stochastic Euler-Maruyama method, strong order 1/2.
    *** For f and g that take arguments in the order x,t rather than t,x.
    """
    dt = t2 - t1
    D = len(x)
    Nrand = len(sidx)

    if fstim is None:
        pf_in_1 = pf
    else:
        try:
            s1 = fstim(t1)
        except TypeError:
            s1 = fstim
        pf_in_1 = (pf, s1)

    xi = np.sqrt(dt) * np.random.randn(Nrand)
    dx = f(x, t1, pf_in_1) * dt
    dx[sidx] += g(x, t1, pg) * xi

    return dx
