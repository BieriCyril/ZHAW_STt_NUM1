g = 9.81
l = 0.6

import sys
sys.path.append('../numerikODE/')
from odeSolvers import classicRungeKutta
import numpy as np
import matplotlib.pyplot as plt
import time

def f(t,phi):
    return np.array([phi[1], -g/l*np.sin(phi[0])])

def phipendel(phi0):
    t,phi = classicRungeKutta(40,1e-2,np.array([phi0,0.]),f)
    return lambda s: np.interp(s,t,phi[:,0])
