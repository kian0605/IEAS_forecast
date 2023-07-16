#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 20:42:08 2023

@author: kian
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, inv
import warnings
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as SS
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error as mse
warnings.simplefilter(action='ignore', category=FutureWarning)


T0 = [400]
N10 = [20,80]
N20 = [20,80]
r10 = [5,40]
r20 = [5,40]
sig_f0 = [1]

N_t = len(T0)
N_n1 = len(N10)
N_n2 = len(N20)
N_r1 = len(r10)
N_r2 = len(r20)
N_sf = len(sig_f0)

R = 500


for it in range(N_t):
    T = T0[it]
    for in1 in range(N_n1):
        N1 = N10[in1]
        for in2 in range(N_n2):
            N2 = N20[in2]
            for ir1 in range(N_r1):
                r1 = r10[ir1]
                for ir2 in range(N_r2):
                    r2 = r20[ir2]
                    for isf in range(N_sf):
                        sig_f = sig_f0[isf]
                        np.random.seed(2023)
                        N =  N1+N2
                        lam1 = np.random.randn(N2,r1)
                        lam2 = np.random.randn(N,r2)
                        
                        f = np.random.randn(T,r1)/np.sqrt(r1)*sig_f
                        g = np.random.randn(T,r2)/np.sqrt(r2)
                        e = np.random.randn(T,N)
                        X = g@lam2.T+e
                        X[:,N1:] = X[:,N1:]+f@lam1.T
                        gammahat = np.zeros((R,N1+N2,N_t,N_n1,N_n2,N_r1,N_r2,2))
                        
                        for rr in range(R):
                            print(rr)
                            # specification of y
                            v = np.random.randn(T,1)
                            beta = np.array([np.linspace(r1,1,r1)/r1]).T*4
                            y = f@beta+v
                            
                            gamma0 = lam1@inv(lam1.T@lam1)@beta
                            gamma0 = np.r_[np.zeros((N1,1)),gamma0]
                            #X = SS(with_mean=False).fit(X).transform(X)
                            # LPCA
                            rhat = r1+r2
                            u, s, vh = svd(X,full_matrices=False)
                            fhat = u[:,:rhat]
                            
                            [y_train, y_test, X_train, X_test, fhat_train, fhat_test]= tts(y,X,fhat, train_size =0.8, shuffle=False)
                            
                            reg = LassoLarsCV(cv=10, max_iter=10000,max_n_alphas=2000,fit_intercept=True).fit(fhat_train, y_train.flatten())
                            bhat = reg.coef_
                            gammahat1 = vh[:rhat,:].T@inv(np.diag(s[:rhat]))@bhat
             
                            # Lasso

                            reg = LassoLarsCV(cv=10, max_iter=10000,max_n_alphas=2000,fit_intercept=True).fit(X_train, y_train.flatten())
                            gammahat2 = reg.coef_
                            gammahat[rr,:,it,in1,in2,ir1,ir2,0] = gammahat1
                            gammahat[rr,:,it,in1,in2,ir1,ir2,1] = gammahat2
                        figsize = (5, 4)
                        fig1, ax = plt.subplots(1,1, figsize=figsize, dpi=200)
                        ax.boxplot(gammahat[:,:,it,in1,in2,ir1,ir2,0],flierprops={'marker': 'o', 'markersize': 1})                        
                        plt.xticks([N1,N1+N2],[str(N1),str(N1+N2)])   
                        plt.savefig('/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code/simulation/boxplot_LPCA_'+'T'+str(T)+'N1'+str(N1)+'N2'+str(N2)+'r1'+str(r1)+'r2'+str(r2)+'.png')     
                        
                        figsize = (5, 4)
                        fig1, ax = plt.subplots(1,1, figsize=figsize, dpi=200)
                        ax.boxplot(gammahat[:,:,it,in1,in2,ir1,ir2,1],flierprops={'marker': 'o', 'markersize': 1})                            
                        plt.xticks([N1,N1+N2],[str(N1),str(N1+N2)])   
                        plt.savefig('/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code/simulation/boxplot_Lasso_'+'T'+str(T)+'N1'+str(N1)+'N2'+str(N2)+'r1'+str(r1)+'r2'+str(r2)+'.png')   
                          

              
                            
                            
                            
                            
                            