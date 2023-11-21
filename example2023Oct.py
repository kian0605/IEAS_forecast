#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 13:27:24 2023

@author: kian
"""

# 引入 pandas 套件，並使用 pd 為別名(其提供高效能、簡易使用的資料格式(Data Frame)，以達到快速操作及分析資料)
import pandas as pd 
# 引入 numpy 套件，並使用 np 為別名(其重點在於陣列的操作，其所有功能特色都建築在同質且多重維度的 ndarray（N-dimensional array）上)
import numpy as np 
# 做圖用
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False # 使坐標軸刻度表簽正常顯示正負號
import seaborn as sns; sns.set()
#作用是內嵌畫圖，省略掉plt.show()這一步，直接顯示圖像。如果不加這一句的話，我們在畫圖結束之後需要加上plt.show()才可以顯示圖像
%matplotlib inline

# os 負責程序和作業系統之間的交互，可以處理大部分的文件操作
import os 
#os.getcwd() # 可看到當前工作路徑
#os.chdir(r'/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/code') # 更改當前工作到存放有要讀取之function的位置 (所有"\"要變"\\"才行)
os.chdir(r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\經濟預測\code')
from funs import *


#%%
# 資料儲存的路徑
today = date.today()
#result_path = r'C:\Users\ntpu_metrics\Dropbox\RA\Janice\經濟預測\data' 
#graph_path = r'C:\Users\ntpu_metrics\Dropbox\RA\Janice\經濟預測\graph'
result_path = r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\經濟預測\data'
graph_path = r'C:\Users\kian_\Dropbox\NTPU\RA_project\RA\Janice\經濟預測\graph'
#result_path = '/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/data'
#graph_path = '/home/kian/Dropbox/NTPU/RA_project/RA/Janice/經濟預測/graph'

# 台灣總體統計資料之文件檔名稱
URL_txt = result_path+'/tw_URL.txt'
# 讀入已存有的 Taiwan data
tw_data =  pd.read_csv(result_path+'/taiwan_data.csv',index_col=0,parse_dates=True) #如果CSV文件中要設為index的日期字符串為標準格式，則可以直接透過parse_dates=True 確保正確解析日期成為DatetimeIndex
# 將現有 dataframe 帶入 function(tw_URL)，得到是否有較新的台灣資料的訊息並自動使用與存出較新資料
tw_data = tw_URL(tw_data,URL_txt,df_to_csv=[result_path]) 

# 從 dataframe 中，加以計算成實質進、出口之變數
r_im = tw_data['進出口貿易總值(一般貿易制度)_NTD(百萬元)-進口'] /tw_data['進口物價基本分類指數(按HS)(新臺幣計價)_總指數']
r_ex = tw_data['進出口貿易總值(一般貿易制度)_NTD(百萬元)-出口']/tw_data['出口物價基本分類指數(按HS)(新臺幣計價)_總指數']

X = tw_data.drop(['進出口貿易總值(一般貿易制度)_NTD(百萬元)-進口','進出口貿易總值(一般貿易制度)_NTD(百萬元)-出口','進出口貿易總值(一般貿易制度)_USD(百萬美元)-進口','進出口貿易總值(一般貿易制度)_USD(百萬美元)-出口'],axis=1) # 將台灣資料留下解釋變數的部分
# 以內插法處理 nan 值的部分
X = X.interpolate() 
pmi = pd.read_excel(r'https://www.cier.edu.tw/public/data/PMI%20%E6%AD%B7%E5%8F%B2%E8%B3%87%E6%96%99%20(%E5%AD%A3%E7%AF%80%E8%AA%BF%E6%95%B4).xlsx', index_col=[0], header=[0], skiprows=[0]).dropna(how='all')
pmi.index = pd.date_range(pmi.index[0],pmi.index[-1],freq='MS')
X = pd.concat([X,pmi], axis=1)

# 讀入已存有的 fred 資料
fred_data = pd.read_csv(result_path+'/fred_data.csv',index_col=0,parse_dates=True)

# 比較是否有新資料，並且作整理
fred_data = fred_newdata(fred_data,df_to_csv=[result_path]) 
# 內插法處理 nan 值的部分
fred_data = fred_data.interpolate()

# 欲預測的起始時間<表單式填寫 預測期數(H)、年份(forecast_year)、月份(forecast_month)>
H = 12 
forecast_year ='2023'
forecast_month ='10'
forecast_from = datetime.strptime(forecast_year+'-'+forecast_month+'-1','%Y-%m-%d').date()

# 資料最後的時間
tw_data_last = tw_data.index[-1].date()
fred_data_last = fred_data.index[-1].date()
print('***the latest date of the data:***','\nX:',tw_data_last,'\nFred:',fred_data_last)

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

print('***the latest date of the In-sample data:***','\nwithout Fred:',t_process_f,'\nwith Fred:',t_process_f_with_fred)

from funs import *
# 預測目標為進口(im)
im_model = Get_Forecast(r_im,X,fred_data,t_process_f,t_process_f_with_fred,H)
# 預測目標為出口(ex)    
ex_model = Get_Forecast(r_ex,X,fred_data,t_process_f,t_process_f_with_fred,H)

#設更早以前的時間點為in_sample的時間終點
t_process_f = datetime(2023,1,1)  
t_process_f_with_fred = datetime(2023,1,1) 
H = 12 # 設定要預測幾期的結果

# 預測目標為進口(im)
im_model_old = Get_Forecast(r_im,X,fred_data,t_process_f,t_process_f_with_fred,H)
# 預測目標為出口(ex)    
ex_model_old = Get_Forecast(r_ex,X,fred_data,t_process_f,t_process_f_with_fred,H)

# 欲另外繪圖之資料:

start_t = '2021-6-1' # 圖形橫軸起始時間

im_0 = im_model.forecasts['forecast_without_fred_f']['real'][start_t:]
im_1 = im_model.forecasts['forecast_without_fred_f']['forecast(level)_0'][-(H+1):]
im_2 = im_model.forecasts['forecast_without_fred_f']['forecast_0'][-(H+1):]
im_3 = im_model.forecasts['forecast_with_fred_f']['forecast(level)_0'][-(H+1):]
im_4 = im_model.forecasts['forecast_with_fred_f']['forecast_0'][-(H+1):]

ex_0 = ex_model.forecasts['forecast_without_fred_f']['real'][start_t:]
ex_1 = ex_model.forecasts['forecast_without_fred_f']['forecast(level)_0'][-(H+1):]
ex_2 = ex_model.forecasts['forecast_without_fred_f']['forecast_0'][-(H+1):]
ex_3 = ex_model.forecasts['forecast_with_fred_f']['forecast(level)_0'][-(H+1):]
ex_4 = ex_model.forecasts['forecast_with_fred_f']['forecast_0'][-(H+1):]

im_0_old = im_model_old.forecasts['forecast_without_fred_f']['real'][start_t:]
im_1_old = im_model_old.forecasts['forecast_without_fred_f']['forecast(level)_0'][-(H+1):]
im_2_old = im_model_old.forecasts['forecast_without_fred_f']['forecast_0'][-(H+1):]
im_3_old = im_model_old.forecasts['forecast_with_fred_f']['forecast(level)_0'][-(H+1):]
im_4_old = im_model_old.forecasts['forecast_with_fred_f']['forecast_0'][-(H+1):]

ex_0_old = ex_model_old.forecasts['forecast_without_fred_f']['real'][start_t:]
ex_1_old = ex_model_old.forecasts['forecast_without_fred_f']['forecast(level)_0'][-(H+1):]
ex_2_old = ex_model_old.forecasts['forecast_without_fred_f']['forecast_0'][-(H+1):]
ex_3_old = ex_model_old.forecasts['forecast_with_fred_f']['forecast(level)_0'][-(H+1):]
ex_4_old = ex_model_old.forecasts['forecast_with_fred_f']['forecast_0'][-(H+1):]

#修改 matplotlib 默認參數
plt.rcParams['axes.facecolor'] = 'white' # 為多個繪圖設定預設背景色
plt.rcParams['axes.edgecolor'] = 'black' # 為多個繪圖設定預設邊框色


# 單純進口預測 <增加折點數值標籤 > 建立繪圖物件 fig, 大小為 30 * 10 ------------------
def plot_mean(ax,z,year,q):
    d1 = (q-1)*3+1
    d2 = (q-1)*3+2
    d3 = (q-1)*3+3  
    ax.hlines(y=np.mean(z[datetime(year,d1,1):datetime(year,d3,1)]),xmin=datetime(year,d1,1), xmax=datetime(year,d3,1),linewidth=3)
    ax.text(pd.datetime(year,d2,1),np.mean(z[datetime(year,d1,1):datetime(year,d3,1)])+0.002,'%.2f%%' % (100 * np.mean(z[datetime(year,d1,1):datetime(year,d3,1)])),ha='center', va='bottom',fontsize=18)


fig, ax1 = plt.subplots(figsize = (30, 14))
#plt.suptitle('Forecasts (IM)',fontsize=20)
# 設定小圖  (用 plot 繪圖、設定圖例、刻度)
ax1.set_title('im_result', fontsize=26)
ax1.plot(im_0, linewidth = 4, alpha=1, color='dimgray', label ='real' )
ax1.plot(im_2,marker='o', linewidth = 3, alpha=0.8, color='crimson', label ='forecast' )
ax1.plot(im_4,marker='>', linewidth = 3, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data' )

ax1.plot(im_2_old,marker='o', linewidth = 1, alpha=0.8, color='crimson', label ='forecast_old' )
ax1.plot(im_4_old,marker='>', linewidth = 1, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data_old' )
plot_mean(ax1,im_0,2022,1)
plot_mean(ax1,im_0,2022,2)
plot_mean(ax1,im_0,2022,3)
plot_mean(ax1,im_0,2022,4)
plot_mean(ax1,im_0,2023,1)
plot_mean(ax1,im_0,2023,2)
plot_mean(ax1,im_0,2023,3)
ax1.axhline(y=0,linewidth=1)
plt.setp(ax1.get_xticklabels(), fontsize=26)
#寫上折點的數值 (a b分別代表橫軸與眾軸)
for data,color in zip([im_2,im_4],['crimson','indianred']):
    for a,b in zip(data.index[-12:],data):
        ax1.text(a,b+0.002,'%.2f%%' % (100 * b),bbox=dict(facecolor='%s' %color, alpha=0.1),ha='center', va='bottom',fontsize=18)
ax1.legend(loc='upper left', shadow=True ,fontsize=16)
#ax1.set_yticks(np.linspace(-0.110,0.220,10))
plt.setp(ax1.get_yticklabels(), fontsize=26)
plt.grid(True) 

graph_path = os.path.join(graph_path, today.strftime('%Y%m'))
if os.path.exists(graph_path) == False:
    os.makedirs(graph_path)
plt.savefig(graph_path+'/Predicting the future values_im(rate)'+today.strftime('%b')+str(today.day)+'.png')


fig, ax1 = plt.subplots(figsize = (30, 14))
#plt.suptitle('Forecasts (IM)',fontsize=20)
# 設定小圖  (用 plot 繪圖、設定圖例、刻度)
ax1.set_title('im_result', fontsize=26)
ax1.plot(im_0, linewidth = 4, alpha=1, color='dimgray', label ='real' )
ax1.plot(im_1,marker='o', linewidth = 3, alpha=0.8, color='dodgerblue', label ='forecast(level)' )
ax1.plot(im_3,marker='>', linewidth = 3, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data' )

ax1.plot(im_1_old,marker='o', linewidth = 1, alpha=0.8, color='dodgerblue', label ='forecast(level)_old' )
ax1.plot(im_3_old,marker='>', linewidth = 1, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data_old' )
plot_mean(ax1,im_0,2022,1)
plot_mean(ax1,im_0,2022,2)
plot_mean(ax1,im_0,2022,3)
plot_mean(ax1,im_0,2022,4)
plot_mean(ax1,im_0,2023,1)
plot_mean(ax1,im_0,2023,2)
plot_mean(ax1,im_0,2023,3)
ax1.axhline(y=0,linewidth=1)
plt.setp(ax1.get_xticklabels(), fontsize=26)
#寫上折點的數值 (a b分別代表橫軸與眾軸)
for data,color in zip([im_1,im_3],['dodgerblue','royalblue']):
    for a,b in zip(data.index[-12:],data):
        ax1.text(a,b+0.002,'%.2f%%' % (100 * b),bbox=dict(facecolor='%s' %color, alpha=0.1),ha='center', va='bottom',fontsize=18)

ax1.legend(loc='upper left', shadow=True ,fontsize=16)
#ax1.set_yticks(np.linspace(-0.110,0.220,10))
plt.setp(ax1.get_yticklabels(), fontsize=26)
plt.grid(True) 

plt.savefig(graph_path+'/Predicting the future values_im(level)'+today.strftime('%b')+str(today.day)+'.png')



fig, ax1 = plt.subplots(figsize = (30, 14))
#plt.suptitle('Forecasts (IM)',fontsize=20)
# 設定小圖  (用 plot 繪圖、設定圖例、刻度)
ax1.set_title('im_result', fontsize=26)
ax1.plot(im_0, linewidth = 4, alpha=1, color='dimgray', label ='real' )
ax1.plot(im_1,marker='o', linewidth = 3, alpha=0.8, color='dodgerblue', label ='forecast(level)' )
ax1.plot(im_2,marker='o', linewidth = 3, alpha=0.8, color='crimson', label ='forecast' )
ax1.plot(im_3,marker='>', linewidth = 3, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data' )
ax1.plot(im_4,marker='>', linewidth = 3, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data' )

ax1.plot(im_1_old,marker='o', linewidth = 1, alpha=0.8, color='dodgerblue', label ='forecast(level)_old' )
ax1.plot(im_2_old,marker='o', linewidth = 1, alpha=0.8, color='crimson', label ='forecast_old' )
ax1.plot(im_3_old,marker='>', linewidth = 1, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data_old' )
ax1.plot(im_4_old,marker='>', linewidth = 1, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data_old' )
plot_mean(ax1,im_0,2022,1)
plot_mean(ax1,im_0,2022,2)
plot_mean(ax1,im_0,2022,3)
plot_mean(ax1,im_0,2022,4)
plot_mean(ax1,im_0,2023,1)
plot_mean(ax1,im_0,2023,2)
plot_mean(ax1,im_0,2023,3)
ax1.axhline(y=0,linewidth=1)
plt.setp(ax1.get_xticklabels(), fontsize=26)
#寫上折點的數值 (a b分別代表橫軸與眾軸)
#for data,color in zip([im_1,im_2,im_3,im_4],['dodgerblue','crimson','royalblue','indianred']):
#    for a,b in zip(data.index[-12:],data):
#        ax1.text(a,b+0.002,'%.2f%%' % (100 * b),bbox=dict(facecolor='%s' %color, alpha=0.1),ha='center', va='bottom',fontsize=18)

ax1.legend(loc='upper left', shadow=True ,fontsize=16)
#ax1.set_yticks(np.linspace(-0.110,0.220,10))
plt.setp(ax1.get_yticklabels(), fontsize=26)
plt.grid(True) 

plt.savefig(graph_path+'/Predicting the future values_im'+today.strftime('%b')+str(today.day)+'.png')



# 單純出口預測 <增加折點數值標籤 > 建立繪圖物件 fig, 大小為 30 * 10 ------------------
fig, ax2 = plt.subplots(figsize = (30, 14))
#plt.suptitle('Forecasts (EX)',fontsize=20)
# 設定小圖  (用 plot 繪圖、設定圖例、刻度)
ax2.set_title('ex_result', fontsize=26)
ax2.plot(ex_0, linewidth = 4, alpha=1, color='dimgray', label ='real' )
ax2.plot(ex_2,marker='o', linewidth = 4, alpha=0.8, color='crimson', label ='forecast' )
ax2.plot(ex_4,marker='>', linewidth = 4, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data' )

ax2.plot(ex_2_old,marker='o', linewidth = 1, alpha=0.8, color='crimson', label ='forecast_old' )
ax2.plot(ex_4_old,marker='>', linewidth = 1, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data_old' )
plot_mean(ax2,ex_0,2022,1)
plot_mean(ax2,ex_0,2022,2)
plot_mean(ax2,ex_0,2022,3)
plot_mean(ax2,ex_0,2022,4)
plot_mean(ax2,ex_0,2023,1)
plot_mean(ax2,ex_0,2023,2)
plot_mean(ax2,ex_0,2023,3)
ax2.axhline(y=0,linewidth=1)
plt.setp(ax2.get_xticklabels(), fontsize=26)
#寫上折點的數值 (a b分別代表橫軸與眾軸)
for data,color in zip([ex_2,ex_4],['crimson','indianred']):
    for a,b in zip(data.index[-12:],data):
        ax2.text(a,b+0.002,'%.2f%%' % (100 * b),bbox=dict(facecolor='%s' %color, alpha=0.1),ha='center', va='bottom',fontsize=18)
ax2.legend(loc='upper left', shadow=True ,fontsize=16)
#ax2.set_yticks(np.linspace(-0.010,0.220,10))
plt.setp(ax2.get_yticklabels(), fontsize=26)
plt.grid(True) 

plt.savefig(graph_path+'/Predicting the future values_ex(rate)'+today.strftime('%b')+str(today.day)+'.png')


fig, ax2 = plt.subplots(figsize = (30, 14))
#plt.suptitle('Forecasts (EX)',fontsize=20)
# 設定小圖  (用 plot 繪圖、設定圖例、刻度)
ax2.set_title('ex_result', fontsize=26)
ax2.plot(ex_0, linewidth = 4, alpha=1, color='dimgray', label ='real' )
ax2.plot(ex_1,marker='o', linewidth = 4, alpha=0.8, color='dodgerblue', label ='forecast(level)' )
ax2.plot(ex_3,marker='>', linewidth = 4, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data' )

ax2.plot(ex_1_old,marker='o', linewidth = 1, alpha=0.8, color='dodgerblue', label ='forecast(level)_old' )
ax2.plot(ex_3_old,marker='>', linewidth = 1, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data_old' )
plot_mean(ax2,ex_0,2022,1)
plot_mean(ax2,ex_0,2022,2)
plot_mean(ax2,ex_0,2022,3)
plot_mean(ax2,ex_0,2022,4)
plot_mean(ax2,ex_0,2023,1)
plot_mean(ax2,ex_0,2023,2)
plot_mean(ax2,ex_0,2023,3)
ax2.axhline(y=0,linewidth=1)
plt.setp(ax2.get_xticklabels(), fontsize=26)
#寫上折點的數值 (a b分別代表橫軸與眾軸)
for data,color in zip([ex_1,ex_3],['dodgerblue','royalblue']):
    for a,b in zip(data.index[-12:],data):
        ax2.text(a,b+0.002,'%.2f%%' % (100 * b),bbox=dict(facecolor='%s' %color, alpha=0.1),ha='center', va='bottom',fontsize=18)
ax2.legend(loc='upper left', shadow=True ,fontsize=16)
#ax2.set_yticks(np.linspace(-0.010,0.220,10))
plt.setp(ax2.get_yticklabels(), fontsize=26)
plt.grid(True) 

plt.savefig(graph_path+'/Predicting the future values_ex(level)'+today.strftime('%b')+str(today.day)+'.png')



fig, ax2 = plt.subplots(figsize = (30, 14))
#plt.suptitle('Forecasts (EX)',fontsize=20)
# 設定小圖  (用 plot 繪圖、設定圖例、刻度)
ax2.set_title('ex_result', fontsize=26)
ax2.plot(ex_0, linewidth = 4, alpha=1, color='dimgray', label ='real' )
ax2.plot(ex_1,marker='o', linewidth = 4, alpha=0.8, color='dodgerblue', label ='forecast(level)' )
ax2.plot(ex_2,marker='o', linewidth = 4, alpha=0.8, color='crimson', label ='forecast' )
ax2.plot(ex_3,marker='>', linewidth = 4, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data' )
ax2.plot(ex_4,marker='>', linewidth = 4, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data' )

ax2.plot(ex_1_old,marker='o', linewidth = 1, alpha=0.8, color='dodgerblue', label ='forecast(level)_old' )
ax2.plot(ex_2_old,marker='o', linewidth = 1, alpha=0.8, color='crimson', label ='forecast_old' )
ax2.plot(ex_3_old,marker='>', linewidth = 1, alpha=0.8, color='royalblue', linestyle = '--', label ='forecast(level) with fred data_old' )
ax2.plot(ex_4_old,marker='>', linewidth = 1, alpha=0.8, color='indianred', linestyle = '--', label ='forecast with fred data_old' )

plot_mean(ax2,ex_0,2022,1)
plot_mean(ax2,ex_0,2022,2)
plot_mean(ax2,ex_0,2022,3)
plot_mean(ax2,ex_0,2022,4)
plot_mean(ax2,ex_0,2023,1)
plot_mean(ax2,ex_0,2023,2)
plot_mean(ax2,ex_0,2023,3)
ax2.axhline(y=0,linewidth=1)
plt.setp(ax2.get_xticklabels(), fontsize=26)
#寫上折點的數值 (a b分別代表橫軸與眾軸)
#for data,color in zip([ex_1,ex_2,ex_3,ex_4],['dodgerblue','crimson','royalblue','indianred']):
#    for a,b in zip(data.index[-12:],data):
#        ax2.text(a,b+0.002,'%.2f%%' % (100 * b),bbox=dict(facecolor='%s' %color, alpha=0.1),ha='center', va='bottom',fontsize=18)
ax2.legend(loc='upper left', shadow=True ,fontsize=16)
#ax2.set_yticks(np.linspace(-0.010,0.220,10))
plt.setp(ax2.get_yticklabels(), fontsize=26)
plt.grid(True) 

plt.savefig(graph_path+'/Predicting the future values_ex'+today.strftime('%b')+str(today.day)+'.png')

myVars = locals()
from sklearn.metrics import mean_squared_error as mse
s1 = str(np.argmin(np.array([mse(im_0['2023-1-1':'2023-9-1'],im_1_old['2023-1-1':'2023-9-1']),
mse(im_0['2023-1-1':'2023-9-1'],im_2_old['2023-1-1':'2023-9-1']),
mse(im_0['2023-1-1':'2023-9-1'],im_3_old['2023-1-1':'2023-9-1']),
mse(im_0['2023-1-1':'2023-9-1'],im_4_old['2023-1-1':'2023-9-1'])]))+1)
t1 = myVars['im_'+s1].resample('Q').mean().apply(lambda x: format(x, '.2%'))
print('im_'+s1)

s2 = str(np.argmin(np.array([mse(ex_0['2023-1-1':'2023-9-1'],ex_1_old['2023-1-1':'2023-9-1']),
mse(ex_0['2023-1-1':'2023-9-1'],ex_2_old['2023-1-1':'2023-9-1']),
mse(ex_0['2023-1-1':'2023-9-1'],ex_3_old['2023-1-1':'2023-9-1']),
mse(ex_0['2023-1-1':'2023-9-1'],ex_4_old['2023-1-1':'2023-9-1'])]))+1)
t2 = myVars['ex_'+s2].resample('Q').mean().apply(lambda x: format(x, '.2%'))
print('ex_'+s2)

table = pd.concat([t1,t2],axis=1)
table.columns = ['Import','Export']

#print(table.to_latex())
print(pd.concat([ex_0,ex_1,ex_2,ex_3,ex_4],axis=1).resample('Q').mean().style.format('{:.2%}').to_latex())
print(pd.concat([im_0,im_1,im_2,im_3,im_4],axis=1).resample('Q').mean().style.format('{:.2%}').to_latex())


# 預測目標為進口(im)

###------------- 中文字型設定
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
im_model = Get_Forecast(r_im,X,fred_data,t_process_f,t_process_f_with_fred,H)
im_model.figu_heatmap2( 'forecast_without_fred_f','level', 'test', -100, 100,figsize=(30,10), data_type='coef')
im_model.figu_heatmap2( 'forecast_without_fred_f','level', 'test', -500, 500,figsize=(30,10), data_type='contribution')
im_model.to_excel('forecast_without_fred_f','level',result_path)
# 預測目標為出口(ex)    
ex_model = Get_Forecast(r_ex,X,fred_data,t_process_f,t_process_f_with_fred,H)
ex_model.figu_heatmap2( 'forecast_without_fred_f','level', 'test', -100, 100,figsize=(30,10), data_type='coef')
ex_model.figu_heatmap2( 'forecast_without_fred_f','level', 'test', -500, 500,figsize=(30,10), data_type='contribution')
ex_model.to_excel('forecast_without_fred_f','level',result_path)
