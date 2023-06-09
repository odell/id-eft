# -*- coding: utf-8 -*-
import numpy as np

MASS_ELECTRON = 0.510998950 # MeV
MASS_PROTON = 938.272088 / MASS_ELECTRON # m_e
MASS_PION = 139.57 / MASS_ELECTRON
MU = (MASS_PION*MASS_PROTON) / (MASS_PION + MASS_PROTON) # m_e
FACTOR = 1

C4 = 9/2/2
BETA4 = np.sqrt(2*MU*C4)
L_ALPHA = (2*MU*C4)**0.5

HBAR = 1.054572e-34 # J•s
A0 = 5.291772e-11 # m
ME = 9.109384e-31 # kg

A0_MM = -65.0 # Mott-Massey scattering length (a.u.)