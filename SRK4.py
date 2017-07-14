"""
4-stage stochastic Runge-Kutta integration routine, for a set of D first order 
stochastic ODE's.
"""

import numpy as np

# define Butcher tableaus
b1 = np.array([[0.0, 0.0, 0.0, 0.0],
               [-0.7242916356, 0.0, 0.0, 0.0],
               [0.4237353406, -0.1994437050, 0.0, 0.0],
               [-1.578475506, 0.840100343, 1.738375163, 0.0]])
b2 = np.array([[0.0, 0.0, 0.0, 0.0],
               [2.702000410, 0.0, 0.0, 0.0],
               [1.757261649, 0.0, 0.0, 0.0],
               [-2.918524118, 0.0, 0.0, 0.0]])
gam1 = np.array([-0.7800788474, 0.07363768240, 1.486520013, 0.2199211524])
gam2 = np.array([1.693950844, 1.636107882, -3.024009558, -0.3060491602])

def SRK4_xt(f, g, x, t1, t2, pf, pg=None, sidx=None):
    dt = t2 - t1
    D = len(x)
    dx = np.zeros(D)

    K1 = np.copy(x)
    K2 = x + dt*f(K1,t1,pf)/2.0
    K3 = x + dt*f(K2,t1,pf)/2.0
    K4 = x + dt*f(K3,t1,pf)

    if sidx is not None:
        Nnoise = len(sidx)
        g1 = np.random.normal(size=(Nnoise,))
        g2 = np.random.normal(size=(Nnoise,))
        J1 = np.sqrt(dt)*g1
        J10h = 0.5*np.sqrt(dt) * (g1 + g2/np.sqrt(3.0))

        K2[sidx] += (b1[1,0]*J1 + b2[1,0]*J10h)*g(K1,t1,pg)
        K3[sidx] += (b1[2,0]*J1 + b2[2,0]*J10h)*g(K1,t1,pg) + b1[2,1]*J1*g(K2,t1,pg)
        K4[sidx] += (b1[3,0]*J1 + b2[3,0]*J10h)*g(K1,t1,pg) + b1[3,1]*J1*g(K2,t1,pg) \
                    + b1[3,2]*J1*g(K3,t1,pg)

        dx[sidx] += (gam1[0]*g(K1,t1,pg) + gam1[1]*g(K2,t1,pg) + gam1[2]*g(K3,t1,pg) \
                    + gam1[3]*g(K3,t1,pg))*J1 + (gam2[0]*g(K1,t1,pg) + gam2[1]*g(K2,t1,pg) \
                    + gam2[2]*g(K3,t1,pg) + gam2[3]*g(K4,t1,pg))*J10h

    dx += dt*(f(K1,t1,pf)/6.0 + f(K2,t1,pf)/3.0 + f(K3,t1,pf)/3.0 + f(K4,t1,pf)/6.0)

    return dx

def SRK4(f, g, x, t1, t2, pf, pg=None, sidx=[0]):
    dt = t2 - t1
    D = len(x)
    dx = np.zeros(D)

    K1 = np.copy(x)
    K2 = x + dt*f(t1,K1,pf)/2.0
    K3 = x + dt*f(t1,K2,pf)/2.0
    K4 = x + dt*f(t1,K3,pf)

    if sidx is not None:
        Nnoise = len(sidx)
        g1 = np.random.normal(size=(Nnoise,))
        g2 = np.random.normal(size=(Nnoise,))
        J1 = np.sqrt(dt)*g1
        J10h = 0.5*np.sqrt(dt) * (g1 + g2/np.sqrt(3.0))

        K2[sidx] += (b1[1,0]*J1 + b2[1,0]*J10h)*g(t1,K1,pg)
        K3[sidx] += (b1[2,0]*J1 + b2[2,0]*J10h)*g(t1,K1,pg) + b1[2,1]*J1*g(t1,K2,pg)
        K4[sidx] += (b1[3,0]*J1 + b2[3,0]*J10h)*g(t1,K1,pg) + b1[3,1]*J1*g(t1,K2,pg) \
                    + b1[3,2]*J1*g(t1,K3,pg)

        dx[sidx] += (gam1[0]*g(t1,K1,pg) + gam1[1]*g(t1,K2,pg) + gam1[2]*g(t1,K3,pg) \
                    + gam1[3]*g(t1,K3,pg))*J1 + (gam2[0]*g(t1,K1,pg) + gam2[1]*g(t1,K2,pg) \
                    + gam2[2]*g(t1,K3,pg) + gam2[3]*g(t1,K4,pg))*J10h

    dx += dt*(f(t1,K1,pf)/6.0 + f(t1,K2,pf)/3.0 + f(t1,K3,pf)/3.0 + f(t1,K4,pf)/6.0)
    return dx



################################################################################
# QUESTIONABLE CODE
################################################################################
def SRK4_vec(f, g, x, t, pf, pg):
    NT,D = x.shape
    t1,t2 = (t[:-1],t[1:])
    dt = t2 - t1
    dt = np.tile(dt, (D,1)).T

    g1 = np.random.normal(size=(NT-1,D))
    g2 = np.random.normal(size=(NT-1,D))
    J1 = np.sqrt(dt) * g1
    J10h = 0.5*np.sqrt(dt) * (g1 + g2/np.sqrt(3.0))

    x1 = x[:-1]
    K1 = np.copy(x1)
    K2 = x1 + dt*f(K1,t1,pf)/2.0 + (b1[1,0]*J1 + b2[1,0]*J10h)*g(K1,t1,pg)
    K3 = x1 + dt*f(K2,t1,pf)/2.0 + (b1[2,0]*J1 + b2[2,0]*J10h)*g(K1,t1,pg) + b1[2,1]*J1*g(K2,t1,pg)
    K4 = x1 + dt*f(K3,t1,pf) + (b1[3,0]*J1 + b2[3,0]*J10h)*g(K1,t1,pg) + b1[3,1]*J1*g(K2,t1,pg) + b1[3,2]*J1*g(K3,t1,pg)

    dx = dt*(f(K1,t1,pf)/6.0 + f(K2,t1,pf)/3.0 + f(K3,t1,pf)/3.0 + f(K4,t1,pf)/6.0) \
    + (gam1[0]*g(K1,t1,pg) + gam1[1]*g(K2,t1,pg) + gam1[2]*g(K3,t1,pg) + gam1[3]*g(K3,t1,pg))*J1 \
    + (gam2[0]*g(K1,t1,pg) + gam2[1]*g(K2,t1,pg) + gam2[2]*g(K3,t1,pg) + gam2[3]*g(K4,t1,pg))*J10h
    return dx

def SRK4_h2(f, g, x, t1, t2, pf, pg):
    dt = t2 - t1
    D = len(x)

    xi_1 = np.random.normal(size=D)
    zeta_1r = np.random.normal(size=(5,D))
    mu = np.random.normal(size=D)

    J1 = np.sqrt(dt) * xi_1
    rho5 = 1.0/12.0 - 1.0/(2.0*np.pi**2) * np.sum(1.0/np.arange(1.0, 6.0, 1.0)**2)
    rangetile = np.tile(np.arange(1.0, 6.0, 1.0), (4,1)).T
    a10 = -np.sqrt(2.0)/np.pi * np.sum(zeta_1r/rangetile, axis=0) - 2.0 * np.sqrt(rho5) * mu
    J10h = 0.5*np.sqrt(dt) * (xi_1 + a10)

    K1 = np.copy(x)
    K2 = x + dt*f(K1,t1,pf)/2.0 + (b1[1,0]*J1 + b2[1,0]*J10h)*g(K1,t1,pg)
    K3 = x + dt*f(K2,t1,pf)/2.0 + (b1[2,0]*J1 + b2[2,0]*J10h)*g(K1,t1,pg) + b1[2,1]*J1*g(K2,t1,pg)
    K4 = x + dt*f(K3,t1,pf) + (b1[3,0]*J1 + b2[3,0]*J10h)*g(K1,t1,pg) + b1[3,1]*J1*g(K2,t1,pg) + b1[3,2]*J1*g(K3,t1,pg)

    dx = dt*(f(K1,t1,pf)/6.0 + f(K2,t1,pf)/3.0 + f(K3,t1,pf)/3.0 + f(K4,t1,pf)/6.0) \
    + (gam1[0]*g(K1,t1,pg) + gam1[1]*g(K2,t1,pg) + gam1[2]*g(K3,t1,pg) + gam1[3]*g(K3,t1,pg))*J1 \
    + (gam2[0]*g(K1,t1,pg) + gam2[1]*g(K2,t1,pg) + gam2[2]*g(K3,t1,pg) + gam2[3]*g(K4,t1,pg))*J10h
    return dx

def SRK4_h2_1d(f, g, x, t1, t2, pf, pg, sidx=0):
    dt = t2 - t1

    xi_1 = np.random.normal()
    zeta_1r = np.random.normal(size=(5,))
    mu = np.random.normal()

    J1 = np.sqrt(dt) * xi_1

    prange = np.arange(1.0, 6.0, 1.0)
    rho5 = 1.0/12.0 - 1.0/(2.0*np.pi**2) * np.sum(1.0/prange**2)
    a10 = -np.sqrt(2.0)/np.pi * np.sum(zeta_1r/prange) - 2.0 * np.sqrt(rho5) * mu
    J10h = 0.5*np.sqrt(dt) * (xi_1 + a10)

    K1 = np.copy(x)
    K2 = x + dt*f(K1,t1,pf)/2.0
    K3 = x + dt*f(K2,t1,pf)/2.0
    K4 = x + dt*f(K3,t1,pf)

    K2[sidx] += (b1[1,0]*J1 + b2[1,0]*J10h)*g(K1,t1,pg)
    K3[sidx] += (b1[2,0]*J1 + b2[2,0]*J10h)*g(K1,t1,pg) + b1[2,1]*J1*g(K2,t1,pg)
    K4[sidx] += (b1[3,0]*J1 + b2[3,0]*J10h)*g(K1,t1,pg) + b1[3,1]*J1*g(K2,t1,pg) \
                + b1[3,2]*J1*g(K3,t1,pg)

    dx = dt*(f(K1,t1,pf)/6.0 + f(K2,t1,pf)/3.0 + f(K3,t1,pf)/3.0 + f(K4,t1,pf)/6.0)
    dx[sidx] += (gam1[0]*g(K1,t1,pg) + gam1[1]*g(K2,t1,pg) + gam1[2]*g(K3,t1,pg) \
                + gam1[3]*g(K3,t1,pg))*J1 + (gam2[0]*g(K1,t1,pg) + gam2[1]*g(K2,t1,pg) \
                + gam2[2]*g(K3,t1,pg) + gam2[3]*g(K4,t1,pg))*J10h
    return dx

def SRK4_h2_vec(f, g, x, t, pf, pg):
    NT,D = x.shape
    t1,t2 = (t[:-1],t[1:])
    dt = t2 - t1
    dt = np.tile(dt, (D,1)).T

    xi_1 = np.random.normal(size=(NT-1,D))
    zeta_1r = np.random.normal(size=(NT-1,5,D))
    mu = np.random.normal(size=(NT-1,D))

    J1 = np.sqrt(dt) * xi_1
    rho5 = 1.0/12.0 - 1.0/(2.0*np.pi**2) * np.sum(1.0/np.arange(1.0, 6.0, 1.0)**2)
    rangetile = np.resize(np.tile(np.arange(1.0, 5.0, 1.0), (5,1)).T, (NT-1, 5, D))
    a10 = -np.sqrt(2.0)/np.pi * np.sum(zeta_1r/rangetile, axis=1) - 2.0 * np.sqrt(rho5) * mu
    J10h = 0.5*np.sqrt(dt) * (xi_1 + a10)

    x1 = x[:-1]
    K1 = np.copy(x1)
    K2 = x1 + dt*f(K1,t1,pf)/2.0 + (b1[1,0]*J1 + b2[1,0]*J10h)*g(K1,t1,pg)
    K3 = x1 + dt*f(K2,t1,pf)/2.0 + (b1[2,0]*J1 + b2[2,0]*J10h)*g(K1,t1,pg) + b1[2,1]*J1*g(K2,t1,pg)
    K4 = x1 + dt*f(K3,t1,pf) + (b1[3,0]*J1 + b2[3,0]*J10h)*g(K1,t1,pg) + b1[3,1]*J1*g(K2,t1,pg) + b1[3,2]*J1*g(K3,t1,pg)

    dx = dt*(f(K1,t1,pf)/6.0 + f(K2,t1,pf)/3.0 + f(K3,t1,pf)/3.0 + f(K4,t1,pf)/6.0) \
    + (gam1[0]*g(K1,t1,pg) + gam1[1]*g(K2,t1,pg) + gam1[2]*g(K3,t1,pg) + gam1[3]*g(K3,t1,pg))*J1 \
    + (gam2[0]*g(K1,t1,pg) + gam2[1]*g(K2,t1,pg) + gam2[2]*g(K3,t1,pg) + gam2[3]*g(K4,t1,pg))*J10h
    return dx



################################################################################
# OLD OR BAD CODE, POSSIBLY READY FOR DELETION
################################################################################
def SRK4_1d(f, g, x, t1, t2, pf, pg, sidx=(0,)):
    dt = t2 - t1
    D = len(x)
    
    g1 = np.random.normal()
    g2 = np.random.normal()
    J1 = np.sqrt(dt)*g1
    J10h = 0.5*np.sqrt(dt) * (g1 + g2/np.sqrt(3.0))

    K1 = np.copy(x)
    K2 = x + dt*f(K1,t1,pf)/2.0
    K3 = x + dt*f(K2,t1,pf)/2.0
    K4 = x + dt*f(K3,t1,pf)

    K2[sidx] += (b1[1,0]*J1 + b2[1,0]*J10h)*g(K1,t1,pg)
    K3[sidx] += (b1[2,0]*J1 + b2[2,0]*J10h)*g(K1,t1,pg) + b1[2,1]*J1*g(K2,t1,pg)
    K4[sidx] += (b1[3,0]*J1 + b2[3,0]*J10h)*g(K1,t1,pg) + b1[3,1]*J1*g(K2,t1,pg) \
                + b1[3,2]*J1*g(K3,t1,pg)

    dx = dt*(f(K1,t1,pf)/6.0 + f(K2,t1,pf)/3.0 + f(K3,t1,pf)/3.0 + f(K4,t1,pf)/6.0)
    dx[sidx] += (gam1[0]*g(K1,t1,pg) + gam1[1]*g(K2,t1,pg) + gam1[2]*g(K3,t1,pg) \
                + gam1[3]*g(K3,t1,pg))*J1 + (gam2[0]*g(K1,t1,pg) + gam2[1]*g(K2,t1,pg) \
                + gam2[2]*g(K3,t1,pg) + gam2[3]*g(K4,t1,pg))*J10h
    return dx

def SRK4_1d_tx(f, g, x, t1, t2, pf, pg, sidx=0):
    dt = t2 - t1
    D = len(x)

    g1 = np.random.normal()
    g2 = np.random.normal()
    J1 = np.sqrt(dt)*g1
    J10h = 0.5*np.sqrt(dt) * (g1 + g2/np.sqrt(3.0))

    K1 = np.copy(x)
    K2 = x + dt*f(t1,K1,pf)/2.0
    K3 = x + dt*f(t1,K2,pf)/2.0
    K4 = x + dt*f(t1,K3,pf)

    K2[sidx] += (b1[1,0]*J1 + b2[1,0]*J10h)*g(t1,K1,pg)
    K3[sidx] += (b1[2,0]*J1 + b2[2,0]*J10h)*g(t1,K1,pg) + b1[2,1]*J1*g(t1,K2,pg)
    K4[sidx] += (b1[3,0]*J1 + b2[3,0]*J10h)*g(t1,K1,pg) + b1[3,1]*J1*g(t1,K2,pg) \
                + b1[3,2]*J1*g(t1,K3,pg)

    dx = dt*(f(t1,K1,pf)/6.0 + f(t1,K2,pf)/3.0 + f(t1,K3,pf)/3.0 + f(t1,K4,pf)/6.0)
    dx[sidx] += (gam1[0]*g(t1,K1,pg) + gam1[1]*g(t1,K2,pg) + gam1[2]*g(t1,K3,pg) \
                + gam1[3]*g(t1,K3,pg))*J1 + (gam2[0]*g(t1,K1,pg) + gam2[1]*g(t1,K2,pg) \
                + gam2[2]*g(t1,K3,pg) + gam2[3]*g(t1,K4,pg))*J10h
    return dx
