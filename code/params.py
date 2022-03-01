import numpy as np

parameters = {}

default_params = {
	'dA': 0.5*(np.sqrt(2)-1),
	'dB': 0.5*(np.sqrt(2)+1),
	'c': 0.1,
	'sigma': 0.2,
	'Nsteps': int(1e6)
}

Dt = 1./default_params['dA']
Ss = np.array([250,251,252], dtype=int)
Ntaus = np.array([900, 1000, 1100])

i = 0
for S in Ss:
	for Ntau in Ntaus:
		parameters[i] = default_params.copy()
		parameters[i]['Dt'] = Dt
		parameters[i]['S'] = S
		parameters[i]['Ntau'] = Ntau
		i += 1