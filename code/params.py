import numpy as np
import pandas as pd
import os


LOCAL_PATH = os.getcwd()
SIMULATIONS_PATH = '../../simulations'
LOG_PATH = '../../log'
FIG_PATH = '../../fig'

os.makedirs(SIMULATIONS_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
os.makedirs(FIG_PATH, exist_ok=True)

parameters = {}

default_params = {
	'dA': 0.5*(np.sqrt(2)-1),
	'dB': 0.5*(np.sqrt(2)+1),
	'c': 0.25,
	'sigma': .2,
	'Nsteps': int(1e5)
}

Dt = 1e-3/default_params['dA']
Ss = np.array([100, 101], dtype=int)
Ntaus = np.array([900, 950, 1000, 1050, 1100])

i = 0
for S in Ss:
	for Ntau in Ntaus:
		parameters[i] = default_params.copy()
		parameters[i]['Dt'] = Dt
		parameters[i]['S'] = S
		parameters[i]['Ntau'] = Ntau
		parameters[i]['tau'] = Ntau*Dt
		parameters[i]['rB'] = np.sqrt(S*default_params['c'])*default_params['sigma']
		i += 1

pd.DataFrame(parameters).T.to_csv('parameters.csv')