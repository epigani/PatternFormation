#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from scipy.special import lambertw
from scipy import linalg
from scipy.stats import norm

import os
from os.path import isfile, join

import time
import pickle
import gzip
import sys

import params as par

LOCAL_PATH = os.getcwd()
SIMULATIONS_PATH = '../../simulations'
os.makedirs(SIMULATIONS_PATH, exist_ok=True)
LOG_PATH = '../../log'
os.makedirs(LOG_PATH, exist_ok=True)


def return_parameters(params):
    dA = params['dA']
    dB = params['dB']
    S = params['S']
    c = params['c']
    sigma = params['sigma']
    Nsteps = params['Nsteps']
    Ntau = params['Ntau']
    Dt = params['Dt']
    return dA, dB, S, c, sigma, Nsteps, Ntau, Dt

def convert_time(time):
    h = int(time/3600)
    m = int((time-3600*h)/60)
    s = int(time-3600*h-60*m)
    return h, m, s

def write_log(dA, dB, S, c, sigma, Nsteps, Ntau, Dt, elapsed_time):
    h, m, s = convert_time(elapsed_time)
    os.chdir(LOG_PATH)
    t = time.asctime(time.localtime())
    filename = 'log_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}.txt'.format(dA,dB,S,c,sigma,Nsteps,Ntau)
    with open(filename, 'a') as f:
        f.write('\n##### {time} ##### \n'.format(time=t))
        f.write('Analysis of dA = {0:1.2f}, dB = {1:1.2f}, S = {2}, c = {3:1.2f}, sigma = {4:1.2f}, Nsteps = {5}, tau = {6} \n'.format(dA,dB,S,c,sigma,Nsteps,Ntau))
        f.write('Elapsed time: {} h {} min {} s \n \n \n'.format(h, m, s))
    os.chdir(LOCAL_PATH)


def make_B(S, dB, c, sigma):
    B = np.zeros((S,S))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if np.random.rand() < c:
                B[i,j] = np.random.normal(0,sigma)
    np.fill_diagonal(B, -dB)
    return B


def numerical_integration(S, dA, dB, c, sigma, Dt, Nsteps, Ntau, x0):
    B = make_B(S, dB, c, sigma)
    eig_B, P_B = linalg.eig(B)
    P_Binv = linalg.inv(P_B)

    u0 = P_Binv@(x0*np.ones(S))
    U = np.zeros((10*Ntau+Nsteps+1,S), dtype='complex_')
    U[0:10*Ntau] = np.reshape(np.tile(u0, 10*Ntau), (10*Ntau,S))
    X = np.zeros((Nsteps+1,S))
    X[0] = x0*np.ones(S)
    ftau = np.exp(-np.arange(0,10*Ntau,1)/Ntau)[::-1]

    stop_counter = 0
    count_each = 100

    for i in range(count_each):
        for step in range(int(Nsteps/count_each)):
            time_step = int(count_each*i + step)
            # 10*Ntau is a way to discretize the integral, cutting the integration to a meaningful point
            U[10*Ntau+time_step+1] = U[10*Ntau+time_step]*(1-dA*Dt)+Dt/Ntau*eig_B*(U[time_step:10*Ntau+time_step].T@ftau).T*(1+0*1j)
            xf = P_B@U[Ntau+step+1]
            X[time_step+1] = xf.real
        if (np.abs(X[time_step+1])< 1e-6).all():
            stop_counter += 1
            if stop_counter >= 5:
                break
        print ('{} percent of the simulation done'.format(100*i/count_each))


    os.chdir(SIMULATIONS_PATH)
    for i in range(100):
        filename = 'Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.npz'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i)
        if os.path.exists(filename)==False:
            break
    np.savez_compressed(filename, X=X, U=U, S=S, dA=dA, dB=dB, c=c, sigma=sigma, Dt=Dt, Nsteps=Nsteps, Ntau=Ntau, x0=x0)
    os.chdir(LOCAL_PATH)
    return X, U



def main():
    start_time = time.time()
    dA, dB, S, c, sigma, Nsteps, Ntau, Dt = return_parameters(par.parameters[float(sys.argv[1])])
    # for key in par.parameters:
    #     print (key,par.parameters[key])
    print('Analysis of dA = {0:1.2f}, dB = {1:1.2f}, S = {2}, c = {3:1.2f}, sigma = {4:1.2f}, Nsteps = {5}, Ntau = {6} ongoing'.format(dA,dB,S,c,sigma,Nsteps,Ntau))
    X, U = numerical_integration(S=S, dA=dA, dB=dB, c=c, sigma=sigma, Dt=Dt, Nsteps=Nsteps, Ntau=Ntau, x0=1)
    elapsed_time = time.time() - start_time
    write_log(dA, dB, S, c, sigma, Nsteps, Ntau, Dt, elapsed_time)
    #h, m, s = convert_time(elapsed_time)



if __name__ == '__main__':
    main()