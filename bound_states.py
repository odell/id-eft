'''
Coordinate-space functions for computing binding energies and wave functions.
'''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from utility import log_mesh
from constants import MU

IVP_R_ENDPTS = np.array([1e-2, 100])

def wave_function_solution(v_r, energy, r_endpts=IVP_R_ENDPTS):
    sol = solve_ivp(
        lambda r, phi: np.array([phi[1], 2*MU*(v_r(r) - energy) * phi[0]], dtype=object),
        r_endpts, [r_endpts[0], 1], rtol=1e-8, atol=1e-12,
        dense_output=True, method='DOP853'
    )
    return sol.sol


def D(energy, interaction, r_0, endpts=IVP_R_ENDPTS):
    sol = wave_function_solution(interaction, energy, r_endpts=[IVP_R_ENDPTS[0], 1.1*r_0])
    u, up = sol(r_0)
    gamma = np.sqrt(2*MU*-energy)
    return gamma*u + up
    
    
def bound_state(interaction, guess, r_0, endpts=IVP_R_ENDPTS):
    result = fsolve(lambda en: D(en, interaction, r_0, endpts=endpts), guess, factor=0.1)
    return result[0]


def normalization_r_mesh(r_max):
    return log_mesh(0, r_max, 5000)


def normalized_wave_function(v_r, energy, r_max=IVP_R_ENDPTS[1]):
    r, wr = normalization_r_mesh(r_max)
    u = wave_function_solution(v_r, energy)(r)[0]
    iC2 = np.dot(wr, u**2)
    return 1/np.sqrt(iC2) * u