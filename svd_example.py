# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

T0 = [50,100,200,400]
N10 = [20,40,80,160]
N20 = [20,40,80,160]
r10 = [5,10,20,40]
r20 = [5,10,20,40]
sig_f0 = [1,2]

N_t = len(T0)
N_n1 = len(N10)
N_n2 = len(N20)
N_r1 = len(r10)
N_r2 = len(r20)
N_sf = len(sig_f0)

R = 500
MSE = np.zeros((R,N_t,N_n1,N_n2,N_r1,N_r2,N_sf,2))

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
                        lam1 = np.random.randn(N2,r1)+1
                        lam2 = np.random.randn(N,r2)+1
                        
                        f = np.random.randn(T,r1)/np.sqrt(r1)*sig_f
                        g = np.random.randn(T,r2)/np.sqrt(r2)
                        e = np.random.randn(T,N)
                        X = g@lam2.T+e
                        X[:,N1:] = X[:,N1:]+f@lam1.T
                        
                        for rr in range(R):
                            # specification of y
                            v = np.random.randn(T,1)
                            beta = np.array([np.linspace(r1,1,r1)/r1]).T*2
                            y = f@beta+v
                            

                            #X = SS(with_mean=False).fit(X).transform(X)
                            # LPCA
                            rhat = r1+r2
                            u, s, vh = svd(X,full_matrices=False)
                            fhat = u[:,:rhat]
                            
                            [y_train, y_test, X_train, X_test, fhat_train, fhat_test]= tts(y,X,fhat, train_size =0.8, shuffle=False)
                            
                            reg = LassoLarsCV(cv=10, max_iter=10000,max_n_alphas=2000,fit_intercept=True).fit(fhat_train, y_train.flatten())
                            bhat = reg.coef_
                            gammahat = vh[:rhat,:].T@inv(np.diag(s[:rhat]))@bhat
                            yhat = X_test@gammahat 
                            #plt.plot(gammahat)
                            #np.corrcoef(np.c_[y_test,yhat[np.newaxis].T].T)
                            
                            # Lasso

                            reg = LassoLarsCV(cv=10, max_iter=10000,max_n_alphas=2000,fit_intercept=True).fit(X_train, y_train.flatten())
                            #reg.coef_
                            yhat2 = reg.predict(X_test)[np.newaxis].T
                            #plt.plot(np.c_[y_test,yhat[np.newaxis].T,yhat2])
                            #plt.legend(['y_true','LPCA','Lasso'], loc=0)
                            MSE[rr,it,in1,in2,ir1,ir2,isf,0] = mse(y_test,yhat)
                            MSE[rr,it,in1,in2,ir1,ir2,isf,1] = mse(y_test,yhat2)
import pickle

with open('/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code/sim_result.pickle', 'wb') as f:
    pickle.dump(MSE, f)
    
    
with open('/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code/sim_result.pickle','rb') as f:
    MSE = pickle.load(f)    
    
MSE = np.mean(MSE,axis=0)

for ir1 in range(N_r1):
    r1 = r10[ir1]
    for ir2 in range(N_r2):
        r2 = r20[ir2]
        MSEtable=pd.DataFrame()
        for it in range(N_t):
            T = T0[it]
            MSEtable = pd.concat([MSEtable,pd.DataFrame(MSE[it,:,:,ir1,ir2,1,0]/MSE[it,:,:,ir1,ir2,1,1])],axis=0)
        print(r1,r2)
        for ii in range(MSEtable.shape[1]):
            MSEtable.iloc[:,ii] = MSEtable.iloc[:,ii].map('{:,.3f}'.format)
        print(MSEtable.to_latex(header=False,index=False))
    
    
    
    
    
    
    
    
    
    
    
    