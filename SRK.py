"""
A collection of stochastic Runge-Kutta integration schemes.
"""

import numpy as np

def milRK_1p5(f, x, t1, t2, pf, g, sidx, pg=None, dgdt=None, fstim=None):
    """
    A strong order 1.5 method outlined in Milstein & Tretyakov,
    "Stochastic Numerics for Mathematical Physics."
    Assumes the noise is *additive*.
    This method requires the time derivative of g if it is not constant.
    It is assumed that the noise is diagonal in the model components, and that
    there is only one process per component.
    """
    dt = t2 - t1
    D = len(x)
    dx = np.zeros(D)

    Nrand = len(sidx)
    q1 = np.sqrt(dt) * np.random.randn(Nrand)
    q2 = np.sqrt(dt) * np.random.randn(Nrand)

    if fstim is None:
        pf_in_1 = pf
        pf_in_2 = pf
    else:
        try:
            s1, s2 = (fstim(t1), fstim(t2))
        except TypeError:
            s1, s2 = fstim
        pf_in_1 = (pf, s1)
        pf_in_2 = (pf, s2)

    termP = np.zeros(D)
    termM = np.zeros(D)
    termP[sidx] += g(t1, pg) * ((0.5 + 1.0/np.sqrt(6.0))*q1 + q2/np.sqrt(12.0))
    termM[sidx] += g(t1, pg) * ((0.5 - 1.0/np.sqrt(6.0))*q1 + q2/np.sqrt(12.0))
    dx[sidx] += g(t1, pg) * q1
    dx += dt/2.0 * f(t1, x + termP, pf_in_1)
    dx += dt/2.0 * f(t2, x + dt*f(t1, x, pf_in_1) + termM, pf_in_2)
    if dgdt is not None:
        dx[sidx] += dgdt(t1, pg) * dt * (q1/2.0 - q2/np.sqrt(12.0))

    return dx

def milRK2_mult(f, g, x_n, Z_n, t_n, t_np1, sidx, pf, pg, sigma_Z, fstim=None):
    """
    An RK method of order (2, 1.5) outlined in Milstein & Tretyakov,
    "Stochastic Numerics for Mathematical Physics."
    This is meant for equations with multiplicative noise of the form
    dX = f(X) dt + g(X) Z dt
    dZ = b dW
    """
    dt = t_np1 - t_n
    D = len(x_n)
    dx = np.zeros(D)
    Nrand = len(sidx)

    if fstim is None:
        pf_stim = pf
    else:
        try:
            s_n = fstim(t_n)
        except TypeError:
            s_n = fstim
        pf_stim = (pf, s_n)

    xi_n = np.random.randn(Nrand)
    eta_n = np.random.randn(Nrand)
    Zbar_n = Z_n + sigma_Z*xi_n*np.sqrt(dt)

    # here, f and g are pre-evaluated for speed
    f_n = f(t_n, x_n, pf_stim)
    xbar_n = x_n + dt * f_n
    g_n = g(t_n, x_n, pg)
    xbar_n[sidx] += dt * g_n*Z_n
    gbar_n = g(t_n, xbar_n, pg)
    fbar_n = f(t_n, xbar_n, pf_stim)

    dx += dt/2.0 * (f_n + fbar_n)
    dx[sidx] += dt/2.0 * (g_n*Z_n + gbar_n*Zbar_n) + \
        g_n * sigma_Z * dt**1.5 * eta_n / np.sqrt(12.0)

    dZ = sigma_Z*xi_n*np.sqrt(dt) + dt/2.0*(Z_n + Zbar_n)

    return dx, dZ

################################################################################
# NOT FINISHED YET
################################################################################
def SRA3(f, x, t1, t2, pf, g=None, pg=None, sidx=None):
    """
    A three-stage method proposed in Roebler (2010), which is strong order 1.5,
    and of overall strong order (3.0, 1.5).
    """
    dt = t2 - t1
    D = len(x)
    dx = np.zeros(D)
