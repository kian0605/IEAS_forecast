#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:52:59 2023

@author: kian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False  # 使坐標軸刻度表簽正常顯示正負號
import seaborn as sns;

sns.set()

import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os

# os.getcwd() # 可看到當前工作路徑
os.chdir(r'/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code')  # 更改當前工作到存放有要讀取之function的位置 (所有"\"要變"\\"才行)
# os.chdir(r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\經濟預測\code')
from funs import *


def threeprfCV(y, X, rmin, rmax, cv, show):
    from threeprf_funs import threePRF
    from sklearn.model_selection import KFold  # import KFold
    from numpy.linalg import inv
    rhat = range(rmin, rmax)
    kf = KFold(n_splits=cv, random_state=None, shuffle=False)
    mse_cv = np.zeros((cv, len(rhat)))
    for rr in rhat:
        for i, (train_index, test_index) in enumerate(kf.split(y)):
            if show == True:
                print(rr, i)
            y_tr = y.iloc[train_index, :]
            X_tr = X.iloc[train_index, :]
            y_te = y.iloc[test_index, :]
            X_te = X.iloc[test_index, :]
            yhat3, ahat, Avar_a, Ga, Avar_y = threePRF(np.array(y_tr), np.array(X_tr), 0, lmax=rr,
                                                       F0=np.random.randn(len(y_tr), rr))
            N = X_tr.shape[1]
            T1 = len(y_te)
            N1 = np.ones((N, 1))
            T2 = np.ones((T1, 1))
            Jt = np.eye(T1) - T2 @ inv(T2.T @ T2) @ T2.T
            yf = np.mean(y_tr, axis=0) + Jt @ X_te @ ahat
            mse_cv[i, rr - 2] = np.mean((yf - np.array(y_te)) ** 2, axis=0)
    rr = np.argmin(np.mean(mse_cv, axis=0)) + 2
    yhat3, ahat, Avar_a, Ga, Avar_y = threePRF(np.array(y), np.array(X), 0, lmax=rr, F0=np.random.randn(len(y), rr))
    return [yhat3, ahat, Avar_a, Ga, Avar_y]


def forecast2(Y, state, X, t_process, t_process2, H, **kwargs):
    from scipy.linalg import block_diag as bd
    from numpy.linalg import inv
    """
    This function is used for out-of-sample prediction through "LassoCV" regression.

    Parameters
    ----------
    Y : dependent variable data
    state : What is the state of the variable when you want to perform regression estimation. Is the'growth rate' or 'level'.
    X : Independent variable data
    t_process : the last time point of the first group of "in-sample"
    t_process2 : the last time point of the last group of "in-sample"
    H : number of periods to forecast

    **kwargs
    "kwargs['fred_data']" is the independent variable data that has been processed

    Returns
    -------
    coef : estimated results of regression coefficients
    T_in_sample : Time range of in-sample in regression estimation
    f :
    result_gr : Contains two columns of DF. The result of merging the true value into the predicted value, and the result of purely only the true value
    out_mse : out-of-sample mean squared error (OOS MSE) of growth rate

    """
    # 基於django的views內已有控制這部分可刪除
    # if t_process > X.index[-1] or t_process2 > X.index[-1]:
    # a = 'Unreasonable time setting !!! The final time point of the independent variable data is ' + str(X.index[-1])
    # sys.exit(a)  # 退出Python程序，exit(0)表示正常退出。當參數非0時，會引發一個SystemExit異常

    # 以選定的狀態來決定是否要將變數轉為年成長率
    if state == 'growth rate':
        X2 = X.pct_change(12, fill_method=None)  # 年成長率(故其較原X少前12期)
        y0 = Y.pct_change(12, fill_method=None)
        cv = 20

    elif state == 'level':
        X2 = X
        y0 = Y
        cv = 30

    X1 = nrmlize(X2)  # 標準化資料 nrmlize 是先前定義出來的函數
    X1 = X1.dropna(how='any', axis=0)

    # Singular value decomposition
    ux, sx, vx = svd(X1, full_matrices=False)
    f = np.sqrt(len(ux)) * ux[:, :10]
    f = pd.DataFrame(f, index=X1.index)

    if ('fred_data' in kwargs) == True:
        fred_data = kwargs['fred_data']
        fred_data = fred_data.dropna(how='any', axis=0)
        fred_data = nrmlize(fred_data)

        uf, sf, vf = svd(fred_data, full_matrices=False)
        f2 = np.sqrt(len(uf)) * uf[:, :10]
        f2 = pd.DataFrame(f2, index=fred_data.index)

        f = pd.concat([f, f2], axis=1)
    # -------------------------------------------------------------------------------------------------------------------------------
    # 建立時間序列的索引，起始點是被解釋變數的資料起始時間、結束時間是所要預測結果的最後一期時間點、平率是月
    t_start = Y.index[0]
    duration = pd.date_range(start=t_start, end=(t_process2 + rd(months=H)), freq='MS')
    duration_len = len(duration)

    # 建立月份虛擬變數(注意~時間點需設置到所有待預測的時間點才行)
    dummy = pd.get_dummies(pd.DataFrame(np.zeros((duration_len, 1)), index=duration).index.month)
    dummy.index = pd.DataFrame(np.zeros((duration_len, 1)), index=duration).index
    dummy = dummy.iloc[:, 1:]  # 剔除第一列，使其作為對照組

    # 建立出預測結果要存放的空間(注意~時間點需設置到所有待預測的時間點才行)
    s_num = len(pd.date_range(start=t_process, end=t_process2, freq='MS'))  # 計算出 in-sample 的組數
    coef = np.zeros((s_num, H, f.shape[1]))  # 待將迴歸係數放入的空間
    if ('fred_data' in kwargs) == True:
        x_coef = np.zeros((s_num, H, X1.shape[1] + fred_data.shape[1]))
        pred2_t = np.zeros((s_num, H, X1.shape[1] + fred_data.shape[1]))  # used to store the product of coefs and data
    else:
        x_coef = np.zeros((s_num, H, X1.shape[1]))
        pred2_t = np.zeros((s_num, H, X1.shape[1]))
    pred_t = pd.DataFrame(np.zeros((duration_len, s_num)),
                          index=duration)  # 待將預測值放入的 DataFrame (可按時間做比較的結果)，之後也可以加入真實值做比較

    T_in_sample = np.zeros((s_num, H, 2),
                           dtype=datetime)  # 待將每次迴歸 in-sample 時間的起始點與終點放入放入的空間 (dtype()：資料型態 要改成可存放時間的 datetime)
    result_gr = pd.DataFrame()  # 建立一個df放可按時間做比較的結果(含預測值與真實值)
    out_mse = np.zeros((s_num, 1))  # 待放入mse的結果
    Y = Y.rename('real')  # 不改就會是與pred_t的第一欄一樣為0
    Y_gr = Y.pct_change(12, fill_method=None)

    # --------- 進行預測 ---------------------------------------------------------------------------------------------------------------
    for tid in range(0, s_num, 1):
        t_a = t_process + rd(months=tid)  # in-sample 的最後時間點
        t_c = t_a + rd(months=1)  # out-of-sample 的起始時間點
        t_d = t_a + rd(months=H)  # out-of-sample 的最後時間點

        for h in range(1, H + 1, 1):
            # 首先，創建一個完整的 YX data
            rawYX_list = [y0, f.shift(h), dummy]
            rawYX = pd.concat(rawYX_list, axis=1)
            rawYX = rawYX.dropna(how='any', axis=0)
            y = rawYX[t_start:t_process + rd(months=tid)].iloc[:, :1]  # 相當於應變數y
            X3 = rawYX[t_start:t_process + rd(months=tid)].iloc[:, 1:]  # 相當於解釋變數x
            reg = LassoCV(cv=cv, alphas=np.linspace(1.5, 0.001, 100), fit_intercept=True, max_iter=7000,
                          random_state=2023).fit(X3,
                                                 np.array(y).flatten())  # max_iter 由1200改成7000 才不會有"ConvergenceWarning"

            # 儲存迴歸結果
            t_b = t_a + rd(months=h)  # out-of-sample 的每個時間點
            coef[tid, h - 1, :] = reg.coef_[:f.shape[1]]
            if ('fred_data' in kwargs) == True:
                x_coef[tid, h - 1, :] = bd(np.sqrt(len(ux)) * vx[:10, :].T, np.sqrt(len(uf)) * vf[:10, :].T) @ bd(
                    inv(np.diag(sx[:10])), inv(np.diag(sf[:10]))) @ reg.coef_[:f.shape[1]]
                pred2_t[tid, h - 1, :] = np.c_[X1[t_a:t_a], fred_data[t_a:t_a]].flatten() * x_coef[tid, h - 1,
                                                                                            :]  # + dummy[t_b:t_b].values@reg.coef_[f.shape[1]:] + reg.intercept_
            else:
                x_coef[tid, h - 1, :] = np.sqrt(len(ux)) * vx[:10, :].T @ inv(np.diag(sx[:10])) @ reg.coef_[:f.shape[1]]
                pred2_t[tid, h - 1, :] = X1[t_a:t_a].values.flatten() * x_coef[tid, h - 1,
                                                                        :]  # + dummy[t_b:t_b].values@reg.coef_[f.shape[1]:] + reg.intercept_
            pred_t[tid][t_b:t_b] = np.c_[f[t_a:t_a], dummy[
                                                     t_b:t_b]] @ reg.coef_ + reg.intercept_  # all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1 and the array at index 1 has size 0
            T_in_sample[tid, h - 1, 0] = rawYX.index[0]
            T_in_sample[tid, h - 1, 1] = rawYX.index[-1]

        if state == 'level':
            pred_t[tid][:t_a] = Y[:t_a]  # (都尚未是成長率)將真實值合入預測值
            pred_t[tid] = pred_t[tid][:t_d].pct_change(12,
                                                       fill_method=None)  # 控制於'out-of-sample 的最後時間點'，否則會有多H期沒意義的結果，並影響mse估計
            col_name = 'forecast(level)_%s' % tid

        if state == 'growth rate':
            pred_t[tid][:t_a] = Y_gr[:t_a]  # (都已是成長率)將真實值合入預測值
            col_name = 'forecast_%s' % tid

        pred_t = pred_t.rename(columns={tid: col_name})  # 改欄位名稱
        for_mse = pd.concat([pred_t[col_name], Y_gr], axis=1)  # 水平合併
        for_mse = for_mse.dropna(how='any',
                                 axis=0)  # 重點是將尾巴時間對齊 (因為'真實值的最後時間點'與'out-of-sample 的最後時間點'， 兩者長短不一定。ex.有未來預測時後者就長於前者；使用滾動式 in-sample 時，很多時候前者長於後者)
        try:
            out_mse[tid] = mse(for_mse['real'][t_c:], for_mse[col_name][t_c:])
        except ValueError:  # 當所有預測值都沒有真實值可以對照時就會出現 ValueError
            out_mse[tid] = np.nan

    result_gr = pd.concat([pred_t, Y_gr], axis=1)  # 水平合併(將真實值合入預測值)

    return [coef, T_in_sample, f, result_gr, out_mse, x_coef, pred2_t]


def forecast3(Y, state, X, t_process, t_process2, H, method, **kwargs):
    from scipy.linalg import block_diag as bd
    from numpy.linalg import inv
    from sklearn.ensemble import GradientBoostingRegressor as GBR
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import RidgeCV, LassoLarsCV
    """
    This function is used for out-of-sample prediction through "LassoCV" regression.

    Parameters
    ----------
    Y : dependent variable data
    state : What is the state of the variable when you want to perform regression estimation. Is the'growth rate' or 'level'.
    X : Independent variable data
    t_process : the last time point of the first group of "in-sample"
    t_process2 : the last time point of the last group of "in-sample"
    H : number of periods to forecast

    **kwargs
    "kwargs['fred_data']" is the independent variable data that has been processed

    Returns
    -------
    coef : estimated results of regression coefficients
    T_in_sample : Time range of in-sample in regression estimation
    f :
    result_gr : Contains two columns of DF. The result of merging the true value into the predicted value, and the result of purely only the true value
    out_mse : out-of-sample mean squared error (OOS MSE) of growth rate

    """

    # 以選定的狀態來決定是否要將變數轉為年成長率
    if state == 'growth rate':
        X2 = X.pct_change(12, fill_method=None)  # 年成長率(故其較原X少前12期)
        y0 = Y.pct_change(12, fill_method=None)

    elif state == 'level':
        X2 = X
        y0 = Y
    X1 = nrmlize(X2)  # 標準化資料 nrmlize 是先前定義出來的函數

    X1 = X1.dropna(how='any', axis=0)
    if ('fred_data' in kwargs) == True:
        fred_data = kwargs['fred_data']
        fred_data = fred_data.dropna(how='any', axis=0)
        fred_data = nrmlize(fred_data)
        X1 = pd.concat([X1, fred_data], axis=1)
    # -------------------------------------------------------------------------------------------------------------------------------
    # 建立時間序列的索引，起始點是被解釋變數的資料起始時間、結束時間是所要預測結果的最後一期時間點、平率是月
    t_start = Y.index[0]
    duration = pd.date_range(start=t_start, end=(t_process2 + rd(months=H)), freq='MS')
    duration_len = len(duration)

    # 建立月份虛擬變數(注意~時間點需設置到所有待預測的時間點才行)
    dummy = pd.get_dummies(pd.DataFrame(np.zeros((duration_len, 1)), index=duration).index.month)
    dummy.index = pd.DataFrame(np.zeros((duration_len, 1)), index=duration).index
    dummy = dummy.iloc[:, 1:]  # 剔除第一列，使其作為對照組

    # 建立出預測結果要存放的空間(注意~時間點需設置到所有待預測的時間點才行)
    s_num = len(pd.date_range(start=t_process, end=t_process2, freq='MS'))  # 計算出 in-sample 的組數

    x_coef = np.zeros((s_num, H, X1.shape[1]))

    pred_t = pd.DataFrame(np.zeros((duration_len, s_num)),
                          index=duration)  # 待將預測值放入的 DataFrame (可按時間做比較的結果)，之後也可以加入真實值做比較

    T_in_sample = np.zeros((s_num, H, 2),
                           dtype=datetime)  # 待將每次迴歸 in-sample 時間的起始點與終點放入放入的空間 (dtype()：資料型態 要改成可存放時間的 datetime)
    out_mse = np.zeros((s_num, 1))  # 待放入mse的結果
    Y = Y.rename('real')  # 不改就會是與pred_t的第一欄一樣為0
    Y_gr = Y.pct_change(12, fill_method=None)

    # --------- 進行預測 ---------------------------------------------------------------------------------------------------------------
    for tid in range(0, s_num, 1):
        t_a = t_process + rd(months=tid)  # in-sample 的最後時間點
        t_c = t_a + rd(months=1)  # out-of-sample 的起始時間點
        t_d = t_a + rd(months=H)  # out-of-sample 的最後時間點
        print(t_a)
        for h in range(1, H + 1, 1):
            # 首先，創建一個完整的 YX data
            rawYX0_list = [y0, X1, dummy]
            rawYX0 = pd.concat(rawYX0_list, axis=1)
            rawYX0 = rawYX0.dropna(how='any', axis=0)
            N = rawYX0.iloc[:, 1:-11].shape[1]
            T1 = len(rawYX0.iloc[:, :1])
            N1 = np.ones((N, 1))
            T2 = np.ones((T1, 1))
            Jt = np.eye(T1) - T2 @ inv(T2.T @ T2) @ T2.T
            Jn = np.eye(N) - N1 @ inv(N1.T @ N1) @ N1.T
            rawYX2 = rawYX0.copy()
            rawYX2.iloc[:, :1] = np.array(Jt @ rawYX0.iloc[:, :1])
            rawYX2.iloc[:, 1:-11] = np.array(Jt @ rawYX0.iloc[:, 1:-11] @ Jn)
            rawYX2.iloc[:, -11:] = np.array(rawYX0.iloc[:, -11:].copy())

            rawYX = pd.concat([rawYX2.iloc[:, :1], rawYX2.iloc[:, 1:-11].shift(h), rawYX2.iloc[:, -11:]], axis=1)
            rawYX = rawYX.dropna(how='any', axis=0)

            y = rawYX[t_start:t_process + rd(months=tid)].iloc[:, :1]  # 相當於應變數y
            X3 = rawYX[t_start:t_process + rd(months=tid)].iloc[:, 1:]  # 相當於解釋變數x
            t_b = t_a + rd(months=h)  # out-of-sample 的每個時間點
            if method == 'Boosting':
                ##  Boosting regression
                param_grid = {
                    'n_estimators': [30, 40, 50, 60, 70, 80, 90, 100],  # Different number of estimators to try
                    'learning_rate': [0.01, 0.1, 0.2],  # You can also search for other hyperparameters
                    # 'max_depth': [3, 4, 5]
                }
                reg = GBR(max_depth=1, random_state=0)
                grid_search = GridSearchCV(estimator=reg, param_grid=param_grid, scoring='neg_mean_squared_error',
                                           cv=10,
                                           n_jobs=-1)
                grid_search.fit(np.array(X3), np.array(y.iloc[:, 0]))
                # 儲存迴歸結果
                x_coef[tid, h - 1, :] = grid_search.best_estimator_.feature_importances_[
                                        :-11]  # remove the importance of seasonality
                pred_t[tid][t_b:t_b] = grid_search.best_estimator_.predict(
                    np.c_[rawYX2[t_a:t_a].iloc[:, 1:-11], dummy[t_b:t_b]]) + np.mean(rawYX0.iloc[:, :1], axis=0).values
            if method == '3PRF':
                ## 3PRF estimation
                [yhat3, ahat, Avar_a, Ga, Avar_y] = threeprfCV(y, X3, 2, 50, 10, False)
                # 儲存迴歸結果
                x_coef[tid, h - 1, :] = ahat[:-11, 0]  # remove the importance of seasonality
                pred_t[tid][t_b:t_b] = np.c_[rawYX2[t_a:t_a].iloc[:, 1:-11], dummy[t_b:t_b]] @ ahat + np.mean(
                    rawYX0.iloc[:, :1], axis=0).values

            if method == 'Ridge':
                reg = RidgeCV(cv=10, fit_intercept=True).fit(np.array(X3), np.array(y.iloc[:, 0]))
                x_coef[tid, h - 1, :] = reg.coef_[:-11]  # remove the importance of seasonality
                pred_t[tid][t_b:t_b] = reg.predict(np.c_[rawYX2[t_a:t_a].iloc[:, 1:-11], dummy[t_b:t_b]]) + np.mean(
                    rawYX0.iloc[:, :1], axis=0).values

            if method == 'LassoLars':
                reg = LassoLarsCV(cv=10, fit_intercept=True, max_iter=20000).fit(np.array(X3), np.array(y.iloc[:, 0]))
                x_coef[tid, h - 1, :] = reg.coef_[:-11]  # remove the importance of seasonality
                pred_t[tid][t_b:t_b] = reg.predict(np.c_[rawYX2[t_a:t_a].iloc[:, 1:-11], dummy[t_b:t_b]]) + np.mean(
                    rawYX0.iloc[:, :1], axis=0).values

            T_in_sample[tid, h - 1, 0] = rawYX.index[0]
            T_in_sample[tid, h - 1, 1] = rawYX.index[-1]

        if state == 'level':
            pred_t[tid][:t_a] = Y[:t_a]  # (都尚未是成長率)將真實值合入預測值
            pred_t[tid] = pred_t[tid][:t_d].pct_change(12,
                                                       fill_method=None)  # 控制於'out-of-sample 的最後時間點'，否則會有多H期沒意義的結果，並影響mse估計
            col_name = 'forecast(level)_%s' % tid

        if state == 'growth rate':
            pred_t[tid][:t_a] = Y_gr[:t_a]  # (都已是成長率)將真實值合入預測值
            col_name = 'forecast_%s' % tid

        pred_t = pred_t.rename(columns={tid: col_name})  # 改欄位名稱
        for_mse = pd.concat([pred_t[col_name], Y_gr], axis=1)  # 水平合併
        for_mse = for_mse.dropna(how='any',
                                 axis=0)  # 重點是將尾巴時間對齊 (因為'真實值的最後時間點'與'out-of-sample 的最後時間點'， 兩者長短不一定。ex.有未來預測時後者就長於前者；使用滾動式 in-sample 時，很多時候前者長於後者)
        try:
            out_mse[tid] = mse(for_mse['real'][t_c:], for_mse[col_name][t_c:])
        except ValueError:  # 當所有預測值都沒有真實值可以對照時就會出現 ValueError
            out_mse[tid] = np.nan

    result_gr = pd.concat([pred_t, Y_gr], axis=1)  # 水平合併(將真實值合入預測值)

    return [T_in_sample, result_gr, out_mse, x_coef]


# 資料儲存的路徑
today = date.today()
# result_path = r'C:\Users\ntpu_metrics\Dropbox\RA\Janice\經濟預測\data'
# graph_path = r'C:\Users\ntpu_metrics\Dropbox\RA\Janice\經濟預測\graph'
# result_path = r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\經濟預測\data'
# graph_path = r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\經濟預測\graph'
result_path = '/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/data'
graph_path = '/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/graph'

# ---------- dealing with data
# 台灣總體統計資料之文件檔名稱
URL_txt = result_path + '/tw_URL.txt'
tw_data = pd.read_csv(result_path + '/taiwan_data.csv', index_col=0,
                      parse_dates=True)  # 如果CSV文件中要設為index的日期字符串為標準格式，則可以直接透過parse_dates=True 確保正確解析日期成為DatetimeIndex
tw_data = tw_URL(tw_data, URL_txt, df_to_csv=[result_path])

r_im = tw_data['進出口貿易總值(一般貿易制度)_NTD(百萬元)-進口'] / tw_data[
    '進口物價基本分類指數(按HS)(新臺幣計價)_總指數']
r_ex = tw_data['進出口貿易總值(一般貿易制度)_NTD(百萬元)-出口'] / tw_data[
    '出口物價基本分類指數(按HS)(新臺幣計價)_總指數']

X = tw_data.drop(['進出口貿易總值(一般貿易制度)_NTD(百萬元)-進口', '進出口貿易總值(一般貿易制度)_NTD(百萬元)-出口',
                  '進出口貿易總值(一般貿易制度)_USD(百萬美元)-進口', '進出口貿易總值(一般貿易制度)_USD(百萬美元)-出口'],
                 axis=1)  # 將台灣資料留下解釋變數的部分
X = X.interpolate()
pmi = pd.read_excel(
    r'https://www.cier.edu.tw/public/data/PMI%20%E6%AD%B7%E5%8F%B2%E8%B3%87%E6%96%99%20(%E5%AD%A3%E7%AF%80%E8%AA%BF%E6%95%B4).xlsx',
    index_col=[0], header=[0], skiprows=[0]).dropna(how='all')
pmi.index = pd.date_range(pmi.index[0], pmi.index[-1], freq='MS')
X = pd.concat([X, pmi], axis=1)

# 讀入已存有的 fred 資料
fred_data = pd.read_csv(result_path + '/fred_data.csv', index_col=0, parse_dates=True)
fred_data = fred_newdata(fred_data, df_to_csv=[result_path])
fred_data = fred_data.interpolate()

# 欲預測的起始時間<表單式填寫 預測期數(H)、年份(forecast_year)、月份(forecast_month)>
H = 12
forecast_year = '2023'
forecast_month = '11'
forecast_from = datetime.strptime(forecast_year + '-' + forecast_month + '-1', '%Y-%m-%d').date()

# 資料最後的時間
tw_data_last = tw_data.index[-1].date()
fred_data_last = fred_data.index[-1].date()
print('***the latest date of the data:***', '\nX:', tw_data_last, '\nFred:', fred_data_last)

# 預測未來時，in_sample的時間終點，基於tw_data_last會大於fred_data_last的特性建立:
if forecast_from <= fred_data_last:  # 舉例:在2022/10/14當下，fred_data_last為8月，則forecast_from為8月時，in-sample需為7月
    t_process_f = forecast_from - rd(months=1)  # in-sample的最後一期時間，為預測起始期的前一期
    t_process_f_with_fred = forecast_from - rd(months=1)
elif (forecast_from > fred_data_last) & (
        forecast_from <= tw_data_last):  # (記得要括號，不然會出錯)輸入的欲預測時間forecast_from為9月，超出fred_data_last8月，則不管怎樣fred_data之in-sample都是fred_data_last8月
    t_process_f = forecast_from - rd(months=1)
    t_process_f_with_fred = fred_data_last  # in-sample 直接等於資料最後時間
else:  # 只會發生在輸入的欲預測時間，超出兩個data_last，則不管怎樣兩個in-sample都是fdata_last
    t_process_f = tw_data_last
    t_process_f_with_fred = fred_data_last

print('***the latest date of the In-sample data:***', '\nwithout Fred:', t_process_f, '\nwith Fred:',
      t_process_f_with_fred)

# small dataset for the largest sample size
methods = ['Boosting', '3PRF', 'Ridge', 'LassoLars']
result_dict = {}
for mm in methods:
    print(mm)
    result = forecast3(r_ex, 'level', X.dropna(axis=1), datetime(2010, 12, 1), datetime(2023, 5, 1), 12, method=mm)
    result_dict[mm] = result

with open('/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code/empirical_result.pickle', 'wb') as f:
    pickle.dump(result_dict, f)
