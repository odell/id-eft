'''
Coordinate-space functions for accurately computing the scattering length for a system with an
attractive 1/r^4 potential.
'''
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

from constants import MU, BETA4

def wave_function(v_r, energy, r_endpts=np.array([1e-4, 2000])):
    '''
    Solves the reduced, radial Schroedinger equation.
    '''
    sol = solve_ivp(
        lambda r, phi: np.array([phi[1], 2*MU*(v_r(r) - energy) * phi[0]], dtype=object),
        r_endpts, [r_endpts[0], 1], rtol=1e-8, atol=1e-12,
        dense_output=True, method='DOP853'
    )
    return sol.sol


def u0_tail(r, b0, b1):
    '''
    Long-range behavior of the wave function (u) at zero energy in the presence
    of a finite-range interaction.
    '''
    return b0 + b1*r


def i4_tail(r, b0, b1, bm1, bm2):
    '''
    Long-range behavior of the wave function (u) at zero energy in the presence
    of an attractive 1/r^4 interaction.
    '''
    return b0 + b1*r + bm1/r + bm2/r**2


def fit_i4_coefficients(u, r, guess=None):
    '''
    Fits the wave function at zero energy to the expected behavior defined in
    i4_tail.
    '''
    pars, cov = curve_fit(i4_tail, r, u, p0=guess)
    sig = np.sqrt(np.diag(cov))
    return pars, sig


def fit_i4_a0(u, r, guess=None):
    pars, sig = fit_i4_coefficients(u, r, guess=guess)

    b0, b1, bm1, bm2 = pars
    a0 = -b0/b1

    ir1 = bm1/b0
    ir1_pred = BETA4**2/(2*a0)
    
    ir2 = bm2/b0
    ir2_pred = -BETA4**2/6

    return a0, ir1/ir1_pred, ir2/ir2_pred, pars