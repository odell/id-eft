'''
Coordinate-space functions for computing accurate phase shifts for an attractive 1/r^4 system.
'''

import numpy as np
from scipy.integrate import solve_ivp

from constants import MU
from free_solutions import phase_shift, phase_shift_interp

def wave_function(v_r, energy, r_endpts=np.array([1e-3, 3000])):
    sol = solve_ivp(
        lambda r, phi: np.array([phi[1], 2*MU*(v_r(r) - energy) * phi[0]], dtype=object),
        r_endpts, [r_endpts[0], 1], rtol=1e-9, atol=1e-12,
        dense_output=True, method='DOP853'
    )
    return sol.sol


def delta_interp(v_r, energy, r_match=300, max_rel_diff=1e-5, factor=1.1, dx=1e-6):
    k = np.sqrt(2*MU*energy)
    sol = wave_function(v_r, energy)
    
    delta_0 = np.random.rand()
    rel_diff = 1
    
    while rel_diff > max_rel_diff:
        r = np.linspace(0.99*r_match, 1.01*r_match, 100)
        rho = k*r
        u, _ = sol(r)
        delta_1 = phase_shift_interp(u, rho, 0, k*r_match, dx=dx).real
        
        rel_diff = np.abs((delta_1 - delta_0)/delta_0)
        delta_0 = delta_1
        r_match *= factor
    
    return delta_1, r_match/factor


def delta_ivp(v_r, energy, r_match=300, max_rel_diff=1e-5, factor=1.1):
    k = np.sqrt(2*MU*energy)
    sol = wave_function(v_r, energy)
    
    delta_0 = np.random.rand()
    rel_diff = 1
    
    while rel_diff > max_rel_diff:
        u, up = sol(r_match)
        delta_1 = phase_shift(u, up, 0, k*r_match).real
        rel_diff = np.abs((delta_1 - delta_0) / delta_0)
        delta_0 = delta_1
        r_match *= factor
    
    return delta_1, r_match/factor