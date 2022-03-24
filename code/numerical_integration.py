#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from scipy import linalg
from scipy.stats import norm
from scipy import signal

import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from tqdm import tqdm

import os
from os.path import isfile, join

import time
import pickle
import gzip
import sys

import params as par

LOCAL_PATH = os.getcwd()
SIMULATIONS_PATH = par.SIMULATIONS_PATH
LOG_PATH = par.LOG_PATH
FIG_PATH = par.FIG_PATH

custom_params = {"axes.spines.left": True, 
                 "axes.spines.right": True,
                 "axes.spines.bottom": True,
                 "axes.spines.top": True}
sns.set_theme(font="Avenir", font_scale=2., style="ticks", rc=custom_params)


def return_parameters(params):
    dA = params['dA']
    dB = params['dB']
    S = params['S']
    c = params['c']
    sigma = params['sigma']
    Nsteps = params['Nsteps']
    Ntau = params['Ntau']
    Dt = params['Dt']
    tau = params['tau']
    return dA, dB, S, c, sigma, Nsteps, Ntau, Dt, tau

def convert_time(time):
    h = int(time/3600)
    m = int((time-3600*h)/60)
    s = int(time-3600*h-60*m)
    return h, m, s

def write_log(dA, dB, S, c, sigma, Nsteps, Ntau, Dt, flag, elapsed_time=0):
    h, m, s = convert_time(elapsed_time)
    os.chdir(LOG_PATH)
    t = time.asctime(time.localtime())
    filename = '{0}_dA_{1:1.2f}_dB_{2:1.2f}_S_{3}_c_{4:1.2f}_sigma_{5:1.2f}_Nsteps_{6}_Ntau_{7}'.format(flag, dA,dB,S,c,sigma,Nsteps,Ntau)
    with open(filename + '.txt', 'a') as f:
        if elapsed_time==0:
            f.write('\n##### {time} ##### \n'.format(time=t))
            f.write('Analysis of dA = {0:1.2f}, dB = {1:1.2f}, S = {2}, c = {3:1.2f}, sigma = {4:1.2f}, Nsteps = {5}, tau = {6} \n'.format(dA,dB,S,c,sigma,Nsteps,Ntau))
        else:
            f.write('Elapsed time: {} h {} min {} s \n \n \n'.format(h, m, s))
    os.chdir(LOCAL_PATH)
    return filename


def make_B(S, dB, c, sigma):
    B = np.zeros((S,S))
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if np.random.rand() < c:
                B[i,j] = np.random.normal(0,sigma)
    np.fill_diagonal(B, -dB)
    return B

def system_eigenvalues(dA, eig_B, Ntau, Dt):
    a, b, tau = -dA, eig_B, Ntau*Dt
    Delta = (1-a*tau)**2+4*tau*(b+a)
    lmbda_p = (a*tau-1+np.sqrt(Delta, dtype='complex'))/(2*tau)
    #lmbda_m = (a*tau-1-np.sqrt(Delta, dtype='complex'))/(2*tau)
    return lmbda_p

def check_a_priori_stability(lmbda):
    ind_max = np.argmax(lmbda.real)
    Lambda, Omega = lmbda[ind_max].real, lmbda[ind_max].imag
    if Lambda<0:
        stability=True
    else:
        stability=False

    return Lambda, Omega, stability

def chech_theoretical_stability(S, dA, dB, c, sigma, Dt, Ntau):
    rB, tau = np.sqrt(S*c)*sigma, Dt*Ntau
    if rB < 2*np.sqrt(dA*dB):
        stability = True
    elif rB > (dA+dB):
        stability = False
    else:
        chi = np.sqrt((dA+dB)**2-rB**2)
        Delta = (dB-chi)**2-dA**2
        tau_inf, tau_sup = ((dB-chi)-np.sqrt(Delta))/dA**2, ((dB-chi)+np.sqrt(Delta))/dA**2
        if ((tau < tau_inf) or (tau > tau_sup)):
            stability = True
        else:
            stability = False
    return stability


def numerical_integration(S, dA, dB, c, sigma, Dt, Nsteps_max, Ntau, x0, flag):
    range_trial = 20
    for trial in range(range_trial):
        B = make_B(S, dB, c, sigma)
        eig_B, P_B = linalg.eig(B)
        P_Binv = linalg.inv(P_B)
        lmbda = system_eigenvalues(dA, eig_B, Ntau, Dt)
        Lambda, Omega, stability = check_a_priori_stability(lmbda)
        stability_theory = chech_theoretical_stability(S, dA, dB, c, sigma, Dt, Ntau)
        if stability_theory == stability:
            break
        if trial == range_trial - 1:
            print ('ATTENTION!!!! PREDICTED STABILITY AND EXPECTED STABILITY NOT IN AGREEMENT')

        #print ('Nsteps and T simulation : {}, {}'.format(Nsteps_max, 10/(Dt*np.abs(Lambda))))
    Nsteps = min(int(10/(Dt*np.abs(Lambda))), Nsteps_max)
    filename = write_log(dA, dB, S, c, sigma, Nsteps_max, Ntau, Dt, flag, elapsed_time=0)

    os.chdir(LOG_PATH)
    with open(filename + '.txt', 'a') as f:
        f.write('Max Re lambda = {0:4.4f}, Omega = {1:4.4f}. --> {2} \n'.format(Lambda,Omega, stability))
    os.chdir(LOCAL_PATH)

    u0 = P_Binv@(x0*np.ones(S))
    U = np.zeros((10*Ntau+Nsteps+1,S), dtype='complex_')
    U[0:10*Ntau+1] = np.reshape(np.tile(u0, 10*Ntau+1), (10*Ntau+1,S))
    X = np.ones((Nsteps+1,S))
    X[0] = x0*np.ones(S)
    ftau = np.exp(-np.arange(0,10*Ntau,1)/Ntau)[::-1]

    stop_counter = 0
    count_each = 100

    for i in tqdm(range(count_each)):
        for step in range(int(Nsteps/count_each)):
            time_step = int(step + i*Nsteps/count_each)
            #print (time_step)
            # 10*Ntau is a way to discretize the integral, cutting the integration to a meaningful point
            if flag == 'normal':
                U[10*Ntau+time_step+1] = U[10*Ntau+time_step]*(1-dA*Dt)+Dt/Ntau*eig_B*(U[time_step:10*Ntau+time_step].T@ftau).T*(1+0*1j)
            elif flag == 'stabilized':
                try:
                    U[10*Ntau+time_step+1] = U[10*Ntau+time_step]*(1-dA*Dt)+Dt/Ntau*eig_B*(U[time_step:10*Ntau+time_step].T@ftau).T*(1+0*1j) - dA*Dt*P_Binv@(X[time_step]**3)
                    #print (U[10*Ntau+time_step+1])
                except RuntimeWarning:
                    print (U[10*Ntau+time_step])
            
            xf = P_B@U[10*Ntau+time_step+1]
            X[time_step+1] = xf.real

        # os.chdir(FIG_PATH)
        # directory = filename + '/evolution/'
        # #directory = 'Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}/evolution/'.format(dA,dB,S,c,sigma,Nsteps,Ntau)
        # os.makedirs(directory,exist_ok=True)
        # fig, ax = plt.subplots(figsize=(16,9))
        # ax.set_xlabel('time steps')
        # ax.set_ylabel('X(t)')
        # for k in range(10):
        #     ax.plot(X[:,k])
        # #filename = 'Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.png'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i+1)
        # fig.savefig(directory + filename + '_{}.png'.format(i+1), dpi=150)
        # plt.close()
        # os.chdir(LOCAL_PATH)

        if ((np.abs(X[time_step+1,:])< 1e-6).all() or (np.abs(X[time_step+1,:])> 1e6).all()):
            stop_counter += 1
            if stop_counter >= 5:
                print('Steady state or diverging state reached. Interrupting simulation.') 
                break
        #print ('{0:2.1f} percent of the simulation done'.format(100.*(i+1)/count_each))


    os.chdir(SIMULATIONS_PATH)
    for i in range(100):
        #filename = 'Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.npz'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i)
        if os.path.exists(filename + '.npz')==False:
            break
    np.savez_compressed(filename, X=X, U=U, S=S, dA=dA, dB=dB, c=c, sigma=sigma, Dt=Dt, Nsteps=Nsteps, Ntau=Ntau, x0=x0, B=B, eig_B=eig_B, lmbda=lmbda)
    os.chdir(LOCAL_PATH)
    
    # os.chdir(FIG_PATH)
    # fig, ax = plt.subplots(figsize=(16,9))
    # for i in range(10):
    #     ax.plot(X[:,i])
    # fig.savefig(filename+'.png')
    # os.chdir(LOCAL_PATH)
    return X, U


def make_gif(dA,dB,S,c,sigma,Nsteps,Ntau):
    #os.chdir(FIG_PATH)
    directory = '/Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}/evolution/'.format(dA,dB,S,c,sigma,Nsteps,Ntau)
    os.chdir(FIG_PATH + directory)
    images = []
    for i in range(1, 100+1):
        filename = 'Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.png'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i+1)
    #for file_name in sorted(os.listdir(dir_path)):
        try:
            if filename.endswith('.png'):
                #file_path = os.path.join(directory, filename)
                images.append(imageio.imread(filename))
        except:
            pass
    os.chdir(LOCAL_PATH)    
    directory = '/Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}/'.format(dA,dB,S,c,sigma,Nsteps,Ntau)
    os.chdir(FIG_PATH + directory)  
    imageio.mimsave('animation.gif', images, fps=10)
    os.chdir(LOCAL_PATH)


def temporal_evolution_plot(X, dA,dB,S,c,sigma,Nsteps, Ntau, Dt, number_of_species, filename, step=1, unit='time', evolution_of='X'):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_ylabel('{}(t)'.format(evolution_of))
    if evolution_of == 'X':
        points_to_plot = np.arange(0, X.shape[0], step)
    elif evolution_of == 'U':
        try:
            points_to_plot = np.arange(Ntau*10, X.shape[0], step)
        except:
            points_to_plot = np.arange(0, X.shape[0], step)
    if unit == 'time':
        t = points_to_plot*Dt
        ax.set_xlabel('time')
    else:
        t = points_to_plot
        ax.set_xlabel('steps')
    for i in range(min(number_of_species, X.shape[1])):
        #print (X[points_to_plot,i])
        ax.plot(t, X[points_to_plot,i])
        #ax.plot(X[:,i])
    ax.set_title(r'$S={0}, \tilde\tau={1:1.4f}$'.format(S, Ntau*Dt))
    
    #directory = '/Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}/'.format(dA,dB,S,c,sigma,Nsteps,Ntau)
    directory = '/{}/'.format(filename)
    os.chdir(FIG_PATH+directory)
    for i in range(100):
        fn = '{}_{}.png'.format(evolution_of, filename)
        #filename = '{8}distr_evolution_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.png'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i, evolution_of)
        if os.path.exists(fn)==False:
            break
    fig.savefig(fn, dpi=200)
    plt.close()
    os.chdir(LOCAL_PATH)
    plt.show()


def power_spectral_density(X, dA,dB,S,c,sigma,Nsteps, Ntau, Dt, filename, evolution_of):
    fig, ax = plt.subplots(figsize=(16,9))
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('PSD of {}(t)'.format(evolution_of))
    ax.set_title(r'$S={0}, \tilde\tau={1:1.4f}$'.format(S, Ntau*Dt))

    Ps = 0
    for i in range(S):
        f, Pxx_den = signal.welch(X[:,i].real)
        #ax.semilogy(f, Pxx_den)
        try:
            Ps = Ps + Pxx_den
        except:
            Ps = Pxx_den
    ax.semilogy(f, Ps/Ps.sum(), lw=4)

    #directory = '/Xdistr_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}/'.format(dA,dB,S,c,sigma,Nsteps,Ntau)
    directory = '/{}/'.format(filename)
    os.chdir(FIG_PATH+directory)
    for i in range(100):
        fn = 'PowerSpectrum_{}_{}.png'.format(evolution_of, filename)
        #filename = 'Xdistr_PS_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.png'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i)
        if os.path.exists(fn)==False:
            break
    fig.savefig(fn, dpi=200)
    plt.close()
    os.chdir(LOCAL_PATH)

    os.chdir(SIMULATIONS_PATH)
    for i in range(100):
        fn = 'PowerSpectrum_{}_{}.png'.format(evolution_of, filename)
        #filename = 'Xdistr_PS_dA_{0:1.2f}_dB_{1:1.2f}_S_{2}_c_{3:1.2f}_sigma_{4:1.2f}_Nsteps_{5}_tau_{6}_{7}.npz'.format(dA,dB,S,c,sigma,Nsteps,Ntau, i)
        if os.path.exists(fn)==False:
            break
    np.savez_compressed(fn, f=f, Ps=Ps, S=S, dA=dA, dB=dB, c=c, sigma=sigma, Dt=Dt, Nsteps=Nsteps, Ntau=Ntau)
    os.chdir(LOCAL_PATH)

    return f, Ps


def power_spectral_density2(X, dA,dB,S,c,sigma,Nsteps, Ntau, Dt, filename, evolution_of):
    fig, ax = plt.subplots(figsize=(16,9))
    f, Ps = [], 0
    for i in range(S):
        ps, freq = ax.psd(X[:,i], len(X[:,i]), 1)
        f.append(freq[np.argmax(ps)])
        try:
            Ps += ps
        except:
            Ps = ps
        
        #ax.scatter(freq, np.log10(ps))
    f = np.array(f)
    ax.set_xscale('log')
    ax.set_title(r'$S={0}, \tilde\tau={1:1.4f}$'.format(S, Ntau*Dt))


    return fig, ax, f


def RSA(X, kind, S, Ntau, Dt, filename, bins=10, **kwargs):
    xlabels = {'mean':r'$E(x)$', 'mean_squared':r'$\sqrt{E(x^2)}$', 'max':'max(x)', 'normal':'x'}
    if kind == 'mean':
        tmin, tmax =  kwargs['steps_range']
        y, bins = np.histogram(X[tmin:min(tmax,X.shape[0]),:].mean(axis=0), bins=bins)
    elif kind == 'mean_squared':
        tmin, tmax =  kwargs['steps_range']
        y, bins = np.histogram(np.sqrt((X[tmin:min(tmax,X.shape[0]),:]**2).mean(axis=0)), bins=bins)
    elif kind == 'max':
        tmin, tmax =  kwargs['steps_range']
        y, bins = np.histogram((X[tmin:min(tmax,X.shape[0]),:]).max(axis=0), bins=bins)
    elif kind == 'normal':
        t = kwargs['step']
        y, bins = np.histogram(X[t,:], bins=bins)

    fig, ax = plt.subplots(figsize=(16,9))
    x = bins[:-1]
    widths = bins[1:]-bins[:-1]
    ax.bar(x,y,widths,align='edge')
    ax.set_xlabel(xlabels[kind])
    ax.set_ylabel('Number of species')
    ax.set_title(r'$S={0}, \tilde\tau={1:1.4f}$'.format(S, Ntau*Dt))
    directory = '/{}/'.format(filename)
    os.chdir(FIG_PATH + directory)
    fig.savefig('RSA_{}_{}.png'.format(kind, filename), dpi=200)
    plt.close()
    os.chdir(LOCAL_PATH)
    return y, bins
    


def main():
    start_time = time.time()
    dA, dB, S, c, sigma, Nsteps, Ntau, Dt, tau = return_parameters(par.parameters[float(sys.argv[1])])
    flag = sys.argv[2] # can be either 'normal' or 'stabilized' 
    #filename = write_log(dA, dB, S, c, sigma, Nsteps, Ntau, Dt, flag, elapsed_time=0)
    # for key in par.parameters:
    #     print (key,par.parameters[key])
    print('Analysis of dA = {0:1.2f}, dB = {1:1.2f}, S = {2}, c = {3:1.2f}, sigma = {4:1.2f}, Nsteps_max = {5}, Ntau = {6}, Dt = {7:1.4f}, tau = {8:1.4f} ongoing'.format(dA,dB,S,c,sigma,Nsteps,Ntau, Dt, tau))
    X, U = numerical_integration(S=S, dA=dA, dB=dB, c=c, sigma=sigma, Dt=Dt, Nsteps_max=Nsteps, Ntau=Ntau, x0=1, flag=flag)
    elapsed_time = time.time() - start_time
    write_log(dA, dB, S, c, sigma, Nsteps, Ntau, Dt, flag, elapsed_time)
    #h, m, s = convert_time(elapsed_time)


if __name__ == '__main__':
    main()