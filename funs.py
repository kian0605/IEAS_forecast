# -*- coding: utf-8 -*-
# 呼叫網站的 package
import requests 
# 解析 json file (Java Script 輕量資料結構)
import json 
# 引入 pandas 套件，並使用 pd 為別名(其提供高效能、簡易使用的資料格式(Data Frame)，以達到快速操作及分析資料)
import pandas as pd 
#表示日期的類，常用的屬性有year、month、day
from datetime import date
# 引入 numpy 套件，並使用 np 為別名(其重點在於陣列的操作，其所有功能特色都建築在同質且多重維度的 ndarray（N-dimensional array）上)
import numpy as np 
# 用於日期的操作
import time
from datetime import datetime, timedelta 
# df1 == df2（是逐個元素相比較），而寫 assert_frame_equal 不僅能比較兩個df是否相等，還能告訴你哪裡不一樣
from pandas.testing import assert_frame_equal
# svd是"singular value decomposition"
from numpy.linalg import svd 
# 可以用於計算時間差，支持年、月、日、週、時、分、秒等
from dateutil.relativedelta  import relativedelta as rd 
# LassoCV 為類似 OLS 的估計，它可以組合出好的解釋變數組(包含透過強制一些迴歸係數變為0)，使估計更有效
from sklearn.linear_model import LassoCV, Lasso 

# 3D 圖 & heatmap 熱力圖
import matplotlib as mpl
from matplotlib import cm # Matplotlib有許多內定的顏色，以便為3D空間找到一個好的顏色表示
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# 使坐標軸刻度表簽正常顯示正負號
plt.rcParams['axes.unicode_minus'] = False 
# Seaborn 套件是以 matplotlib 為基礎建構的高階繪圖套件，讓使用者更加輕鬆地建立圖表~ 而此用於熱力圖
import seaborn as sns; sns.set() 


# MissingSchema,InvalidSchema 是在 requests 時遇到不對的網址所會產生的 errers
from requests.exceptions import MissingSchema,InvalidSchema
# mse 計算用
from sklearn.metrics import mean_squared_error as mse
# sys 內含很多函數方法和變量，用來處理Python運行時的配置以及資源
import sys 

# urllib.request是一個用來從URLs (Uniform Resource Locators)取得資料的Python模組
from urllib import request
# 匯入Python SSL處理模組 (ssl用於在瀏覽器或用戶的計算機與服務器或網站之間建立加密連接)
import ssl
#表示日期的類，常用的屬性有year、month、day

from sqlalchemy import create_engine
import plotly.graph_objects as go

# dataframe寫入mysql -----------------------------------------------------------------------------
def df_to_mysql(df, tab_name, db_name, password, host='localhost',  \
                user='root', port='3306', encoding='utf8', if_exists='replace', index=True):
    """
    # 需要確定執行環境上有pymysql
    
    dataframe寫入mysql,並且設定索引為Primary Key用來獨一無二地確認一個表格中的每一行資料。
    
    Parameters
    ----------    
    df:需要寫入mysql的dataframe
    tab_name:寫入資料的表名稱
    db_name:需要使用的資料庫名稱
    password:mysql連線密碼
    
    host:mysql連線ip,預設localhost
    user:使用者名稱,預設root              
    port:埠,預設3306
    encoding:字元編碼,預設utf8
    if_exists：表如果存在怎麼處理,預設replace表示刪除原表,建立新表再新增 (其他選項:append表示追加、fail表示取消執行)
    index: 是否包含索引,預設 index=True 表示插入索引,請注意!索引型態不能是字串,不然匯入sql會出錯
    
    Returns
    -------none
    """        
    sql_engine = create_engine(r'mysql+pymysql://{}:{}@{}:{}/{}?charset={}'.format(user,password,host,port,db_name,encoding))
    sql_con = sql_engine.connect()#建立連線        
    df.to_sql(name=tab_name, con=sql_con, if_exists=if_exists, index=index) # 寫入dataframe     
    print("----- 該dataframe完成寫入mysql中的[{}]資料庫，並將該表命名為[{}] -----".format(db_name, tab_name))
    if df.index.name == None: #若沒有索引，則在建立主鍵時會出現Attribute "names" are different/n[left]:  ['myunknowncolumn']/n[right]: [None]
        df.index.name = 'index'
    with sql_engine.connect() as sql_con: #使用sql語句在第一列插入主鍵為dataframe之索引
        sql_con.execute( """ALTER TABLE `{}`.`{}` \
                            CHANGE COLUMN `{}` `{}` DATETIME NOT NULL , \
                            ADD PRIMARY KEY (`{}`);"""
                    .format(db_name, tab_name, df.index.name, df.index.name, df.index.name))
# 標準化 -----------------------------------------------------------------------------
def nrmlize(x): 
    xmean = np.mean(x,0) #算平均，逗號後的 0 是表達" axis = 0：壓縮行，對各列求均值，返回 1 * n 矩陣 (是最常用的，相當於對各欄變數取值)"
    xmad=np.mean((x-xmean)**2,0) #算變異數
    z = (x-xmean)/np.sqrt(xmad) #標準化(分母是標準差)
    return z #設定此 def 最終要回傳數值

# date transformation 民國 to 西元 -----------------------------------------------------------------------------
def date_tf(x): 
    start = '年'
    end = '月'
    y = date(np.int(x[:x.find(start)])+1911,np.int(x[x.find(start)+len(start):x.rfind(end)]),1)
    return y

# 獲取並整理網站資料 -----------------------------------------------------------------------------
def response(request_url):
    # 得到網站資料
    response = requests.get(request_url) 
    # 將文字擷取下來
    response_text = response.text 
    # 以 Json 解析
    today_json = json.loads(response_text) 
    # 取出所要的資料內容
    data = pd.DataFrame(today_json['orgdata']).T 
    # 取出並轉換時間軸為 index
    data.index = pd.date_range(start = date_tf(today_json['row'][0][0]), end = date_tf(today_json['row'][-1][0]), freq='MS') 
    # 建立每一欄位所應對應之名稱
    try:
        data.columns = [today_json['name'] +'_'+ i for i in today_json['colh'][0]] 
    except ValueError:
        data.columns = [today_json['name'] +'_'+today_json['colh'][0][0]+'-'+i for i in today_json['colh'][1]] 
    # 最後得到所要的完整 DataFrame
    return data

# 台灣資料 -----------------------------------------------------------------------------
    
def tw_URL(tw_data, URL_txt, oldest_date='8001',  \
           latest_date=str(int(time.strftime("%Y"))-1911)+time.strftime("%m"), **kwargs):
    """  
    This function will automatically download the latest Fred_data and return "Compared with the original data, is there any data update" and the preliminary data processing result.
    
    Parameters
    ----------
    tw_data = original Taiwan data
    URL_txt = This is the path of the text file of "Taiwan Data Website"
    oldest_date = You want the start time of the Taiwan database URL (format: lunar calendar year and month).The default is '8001'.
    latest_date = You want the latest time of the Taiwan database URL (format: lunar calendar year and month).Default is current time.
    
    **kwargs
    "kwargs['df_to_mysql']" : Write the records stored in the DataFrame to the SQL database using the previously set function.The list needs to be filled, the content is ['tab_name','db_name','password'].
    "kwargs['df_to_csv']" : Save the DataFrame as CSV.Variables that need to fill in the result_path path.
    "kwargs['result_path']"
    
    Returns
    -------
    tw_newdata = Taiwan's latest data
      
    """
    # 首先，開啟存放台灣資料網址的文字檔
    txt = open(URL_txt,encoding='utf-8-sig')  # encoding='utf-8-sig'解決之後逐行讀取到中文的問題(不加會有問題)
    tw_newdata = pd.DataFrame() # 建立一個df放新資料
    for line in txt.readlines(): #逐行讀取開啟的文字檔
        try:
            # 使用先前定義出的 "response" function
            # 另外替換網址中不能解讀的字串為變數
            tw_newdata = tw_newdata.join(response(line.replace("'+oldest_date+'",oldest_date).replace("'+latest_date+'",latest_date).strip('\n')),how='outer')
            print(line)     
        except (MissingSchema,InvalidSchema):
            print('err:',line)            
    txt.close # 關閉一個已打開的文件，且關閉後的文件不能再進行讀寫操作
    tw_newdata.replace(0,np.nan,inplace=True) # 將 0 替換成遺漏值，以方便做比較 
    tw_newdata = tw_newdata.dropna(how='all',axis=0)  # 將全為遺漏值之時間資料整行刪除
    try:    
        # 比較兩個df是否相等
        assert_frame_equal(tw_data,tw_newdata, check_like = True ) 
        # check_like bool，默認為False 如果為 True，則忽略索引和列的順序。注意：索引標籤必須匹配它們各自的行（與列相同）——相同的標籤必須具有相同的數據。
        print('Is there any data update: No')
    except AssertionError:
        print('Is there any data update: Yes')
        # 存出有更新的資料，其中 index=False 
        if ('df_to_csv' in kwargs) == True:
            result_path = kwargs['df_to_csv'][0]
            tw_newdata.to_csv(result_path+'/taiwan_data.csv',index=True, encoding='utf-8-sig') # encoding='utf-8-sig' 解決存出中文亂碼的問題
        if ('df_to_mysql' in kwargs) == True:
            df_to_mysql(df=tw_newdata, tab_name=kwargs['df_to_mysql'][0], db_name=kwargs['df_to_mysql'][1], password=kwargs['df_to_mysql'][2])

    return(tw_newdata)

# 財政部貿易統計資料 -----------------------------------------------------------------------------
def trade_tb(im_ex,oldest_date,latest_date):  
    """
    This function automatically downloads the "Country/Region and Major Commodities Cross Table" for import trade or export trade
and Commodity Cross-Classification Table
    Parameters
    ----------  
    im_ex = Used to indicate whether to choose import trade or export trade. (The string "im" means import and the string "ex" means export.)
    oldest_date = You want the start time of the Taiwan database URL (format: lunar calendar year and month)
    latest_date = You want the latest time of the Taiwan database URL (format: lunar calendar year and month)

    Returns
    -------    
    trade_crosstab = "Country/Region and Major Commodities Cross Table"
    
    """
   
    if im_ex == 'im':
        url='https://web02.mof.gov.tw/njswww/webMain.aspx?sys=220&ym='+oldest_date+'&ymt='+latest_date+'&kind=21&type=1&funid=i8201&cycle=1&outmode=0&compmode=00&outkind=2&fldspc=3,1,5,4,38,2,46,14,70,1,&codlst0=010100110001010100011110111100100000110100&rdm=R55592'    
    elif im_ex == 'ex':
        url='https://web02.mof.gov.tw/njswww/webMain.aspx?sys=220&ym='+oldest_date+'&ymt='+latest_date+'&kind=21&type=1&funid=i8202&cycle=1&outmode=0&compmode=00&outkind=2&fldspc=3,1,5,4,38,2,46,14,70,1,&codlst0=01100011010101000111111101001000001101&rdm=R55592'    

    context = ssl._create_unverified_context() #ssl加密認證移除並存為context
    response = request.urlopen(url, context=context)
    html = response.read()    
    table = pd.read_html(html)
      
    trade_crosstab= pd.DataFrame() # 建立一個df 方便整合新資料
    for x in range(1,len(table),1): #不是從0開始的原因是，該0所代表的df並不是資料本身(而是資料產生時間、格式、單位等資訊)
        for i in range(len(table[1])): 
            table[x].iloc[i:i+1,0:1] = date_tf(table[x].iat[i,0]) #民國 to 西元 (不使用iloc或iat等的話，資料一多容易會有SettingWithCopyWarning)
            #SettingWithCopyWarning:凡出現鍊式賦值的情況，pandas都是不能夠確定到底返回的是一個引用還是一個拷貝。所以遇到這種情況就報warning，而解決方式:可以將鍊式透過iloc等拆解成為多步
        table[x] = table[x].set_index(['Unnamed: 1','Unnamed: 0']) #設置MultiIndex ("國家"Unnamed: 1、"時間"Unnamed: 0)
        trade_crosstab = pd.concat([trade_crosstab,table[x]],axis=1) #合併
    trade_crosstab = trade_crosstab.sort_index(level=['Unnamed: 1','Unnamed: 0'], ascending=[True,True]) # 排序，其中 ascending為True時表示"列升序排列"

    trade_crosstab.replace('－',np.nan,inplace=True) #將遺漏值替換成nan
    return trade_crosstab

# Fred 資料 有(1)~(4) 最後一個(4)是新增的，在主程式頁中只要使用(4)即可! -----------------------------------------------------------------------------
# Fred 資料 (1)-----------------------------------------------------------------------------
def update(fred_data,result_path):
    """
    This function will automatically download the latest Fred_data and return "Compared with the original data, is there any data update" and the preliminary data processing result.
    
    Parameters
    ----------
    fred_data : original fred data
    result_path : path to save updated  data

    Returns
    -------
    rawdata : raw data
    tcode : transformation code
    
    """
    fred_newdata = pd.read_csv('https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv')  
    try:    
        # 比較兩個df是否相等
        assert_frame_equal(fred_data,fred_newdata) 
        print('Is there any data update: No')
    except AssertionError:
        print('Is there any data update: Yes')
        # 存出有跟新的資料，其中 index=False 
        fred_newdata.to_csv(result_path+'/fred_data.csv',index=False)
        
    # 不論資料是否更新，都選用最新讀入的進行資料整理(可以減少 code 的行數)
    rawdata =  fred_newdata.iloc[1:,:] # 排除第1行(轉換代碼)，即是所要的原始資料
    rawdata = rawdata.dropna(how='all',axis=0) # 如此會刪除最後一行 (連時間戳記都沒有的那一行)，方能進行下方時間 index 的轉換
    rawdata.index = pd.to_datetime(rawdata['sasdate'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))) # 將特定列進行時間轉換並設為 index
    rawdata = rawdata.drop(['sasdate'],axis=1) # 將使用完畢、多餘的那一列刪除
    tcode =  fred_newdata.iloc[0:1,1:] # 留下轉換代碼
         
    return(rawdata,tcode)

# Fred 資料 (2)-----------------------------------------------------------------------------

def prepare_missing(rawdata,tcode):
    """
    Function that transforms the raw data into stationary form.
    This function transforms raw data based on each series' transformation code.

    Parameters
    ----------
    rawdata : raw data
    tcode : transformation code

    Returns
    -------
    yt : transformed datas 

    """
    # 創建一個與 rawdata 的大小、索引、列名相同之空的 DataFrame，用來放入轉換後的值
    yt = pd.DataFrame(index=rawdata.index,columns=rawdata.columns) 
    # 取出所有的列名稱為一個 list
    a = list(yt.columns.values)                                 
    little = 1e-6 
    def transxf(rawdata_x,tcode_x):  
        """            
        This function transforms a single series (in a column vector)as specified by a given transfromation code.
    
        Parameters
        ----------
        rawdata_x : raw datas for each columns
        tcode : transformation codes for each columns
    
        Returns
        -------
        dum : transformed datas for each columns
    
        """    
        # case 1, Level (i.e. no transformation): x(t)
        if tcode_x == 1: 
            dum = rawdata_x    
        # case 2, First difference: x(t)-x(t-1)
        elif tcode_x == 2:
            # 縱向1階差分
            dum = rawdata_x.diff() 
        # case 3, Second difference: (x(t)-x(t-1))-(x(t-1)-x(t-2))
        elif tcode_x == 3:
            dum = rawdata_x-2*(rawdata_x.shift(1))+rawdata_x.shift(2)
        # case 4, Natural log: ln(x)
        elif tcode_x == 4:
            if (rawdata_x.min() >= little).bool():
                dum = np.log(rawdata_x)
        # case 5, First difference of natural log: ln(x)-ln(x-1)
        elif tcode_x == 5:
            if (rawdata_x.min() > little).bool():
                # 取log後，1階差分
                dum = np.log(rawdata_x).diff() 
        # case 6, Second difference of natural log: (ln(x)-ln(x-1))-(ln(x-1)-ln(x-2))
        elif tcode_x == 6: 
            if (rawdata_x.min() > little).bool():
               rawdata_x2 = np.log(rawdata_x)
               dum = rawdata_x2-2*(rawdata_x2.shift(1))+rawdata_x2.shift(2)
        # case 7, First difference of percent change: (x(t)/x(t-1)-1)-(x(t-1)/x(t-2)-1)
        elif tcode_x == 7: 
            dum = rawdata_x.pct_change().diff() 
        return dum
    # 將每一列進行轉換，並合併起來
    for i in range(0,np.size(rawdata,1),1): 
        # iat 和 iloc 一樣，都是 index-based 的取值方式，不同處在於 iat 用於取單一值
        yt[a[i]] = transxf(rawdata.iloc[:,i:i+1],tcode.iat[0,i])
    return yt   
   
    
# Fred 資料 (3)-----------------------------------------------------------------------------
def remove_outliers(X): 
    """
    Function that removes outliers from the data. 
    A data point x is considered as an outlier if |x-median|>10*interquartile_range. 

    Parameters
    ----------    
    X : dataset 
    
    Returns
    -------
    Y : dataset with outliers replaced with NaN
    n : number of outliers removed from each series
    """
    # np.tile 是將原來資料中每一維度的元素進行 copy
    median_X_mat = np.tile(X.quantile(0.5),(np.size(X,0),1))    
    IQR_mat = np.tile(X.quantile(0.75)-X.quantile(0.25),(np.size(X,0),1))   
    Z = abs(X-median_X_mat)
    outlier = Z>(10*IQR_mat)
    n = outlier.sum()
    # 若對應的 outlier 為 False，會保留原始值；為 True，則以 nan 替換。
    Y = X.mask(outlier) 
    return [Y,n]


# Fred 資料 (4)-----------------------------------------------------------------------------
def fred_newdata(fred_data, **kwargs):
    """
    This function will automatically download the latest Fred_data and return "Compared with the original data, is there any data update" and the preliminary data processing result.
    
    Parameters
    ----------
    fred_data : original fred data for which data processing has been completed. (including removing outliers and transforming data by transforming the code)
    result_path : path to save updated  data
    
    **kwargs
    "kwargs['df_to_mysql']" : Write the records stored in the DataFrame to the SQL database using the previously set function.The list needs to be filled, the content is ['tab_name','db_name','password'].
    "kwargs['df_to_csv']" : Save the DataFrame as CSV.Variables that need to fill in the result_path path.

    Returns
    -------
    fred_newdata : new  fred data for which data processing has been completed.
    
    """
    fred_newdata = pd.read_csv('https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv')  
    rawdata =  fred_newdata.iloc[1:,:] # 排除第1行(轉換代碼)，即是所要的原始資料
    rawdata = rawdata.dropna(how='all',axis=0) # 如此會刪除最後一行 (連時間戳記都沒有的那一行)，方能進行下方時間 index 的轉換
    rawdata.index = pd.to_datetime(rawdata['sasdate'].apply(lambda x: datetime.strptime(x,'%m/%d/%Y'))) # 將特定列進行時間轉換並設為 index
    rawdata = rawdata.drop(['sasdate'],axis=1) # 將使用完畢、多餘的那一列刪除
    tcode =  fred_newdata.iloc[0:1,1:] # 留下轉換代碼
    fred_newdata =  prepare_missing(rawdata,tcode) #放入先前寫好的資料轉換函數 prepare_missing  
    fred_newdata = fred_newdata.iloc[2:,:]# 依照 matlab 指令之作者，將樣本刪除前兩個月，因為某些列已被差分    
    [fred_newdata,n] = remove_outliers(fred_newdata)# 將資料代入先前寫好的函數 remove_outliers ，以取得刪除異常值後的資料以及刪除的異常值數量
    try:    
        # 比較兩個df是否相等
        assert_frame_equal(fred_data,fred_newdata) 
        print('Is there any data update: No')
    except AssertionError:
        print('Is there any data update: Yes')         
        if ('df_to_csv' in kwargs) == True:
            result_path = kwargs['df_to_csv'][0]
            fred_newdata.to_csv(result_path+'/fred_data.csv',index=True)# 存出 fred data
        if ('df_to_mysql' in kwargs) == True:
            df_to_mysql(df=fred_newdata, tab_name=kwargs['df_to_mysql'][0], db_name=kwargs['df_to_mysql'][1], password=kwargs['df_to_mysql'][2])

    # 不論資料是否更新，都選用最新整理好的資料    
    return(fred_newdata)
    
# forecast -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------


def figu_3D(coef,figu_name):
    # 軸坐標
    S=np.arange(1,np.size(coef,2)+1,1) # (10,)
    B=np.arange(1,np.size(coef,0)+1,1) # (133,)
    
    bb, ss=np.meshgrid(B, S) #網格化坐標
    B, S=bb.flatten(), ss.flatten() #矩陣扁平化
    bottom=np.zeros_like(B)#設置柱狀圖的底端位值
    #上方函數主要是想實現構造一個矩陣，其初始化為全0、維度與括號內矩陣一致(方便的構造了新矩陣，無需參數指定shape大小)

    #每一個柱子的長和寬
    width=0.4
    height=0.5
    
    #柱子漸變顏色
    cmap = cm.get_cmap('jet') # 使用colormap，將數據映為顏色模型    
    max_height = np.max(coef) # 標準化顏色值的範圍
    min_height = np.min(coef)    

    for h in range(1,np.size(coef,1)+1,1): 
        Z = coef[:,h-1,:].T  #此時的z會跟coef的"轉置"一樣  
        dz=Z.flatten()        
        rgba = [cmap((k-min_height)/max_height) for k in dz] # 獲得每個 dz 的顏色值
    
        #繪圖設置
        fig=plt.figure(figsize=(24,16)) # figsize 用來設置圖形大小
        ax=fig.gca(projection='3d') # 設定三維
        ax.bar3d(S, B, bottom, width, height, dz, shade=True, color=rgba, edgecolor='None') #柱狀部分 
        #shade:是否顯示陰影（設置為True立體效果會更好）
        #edgecolor:柱狀不分的邊框顏色設置
    
        #坐標軸設置
        ax.set_title(('Forecast  t + %s  period' %h)+' ('+figu_name+')' , fontsize=18)
        ax.set_xlabel('Beta_i',fontsize=12)
        ax.set_ylabel('In-Sample_i',fontsize=12)
        ax.set_zlabel('values of Beta',fontsize=12)
        ax.set(xticks=np.arange(1,np.size(coef,2)+1,1),
               yticks=np.linspace(1,np.size(coef,0),12),
               zticks=np.linspace(Z.min(), Z.max(),12)
              )

        # 調整視角
        ax.view_init(elev=15,    
                     azim=240   
                    )
        plt.show() 


# -----------------------------------------------------------------------------

def forecast(Y, state, X, t_process, t_process2, H, **kwargs):
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
    #if t_process > X.index[-1] or t_process2 > X.index[-1]:
        #a = 'Unreasonable time setting !!! The final time point of the independent variable data is ' + str(X.index[-1])
        #sys.exit(a)  # 退出Python程序，exit(0)表示正常退出。當參數非0時，會引發一個SystemExit異常


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
    ux, sx, vx = svd(X1,full_matrices=False)
    f = np.sqrt(len(ux)) * ux[:, :10]
    f = pd.DataFrame(f, index=X1.index)

    if ('fred_data' in kwargs) == True:
        fred_data = kwargs['fred_data']
        fred_data = fred_data.dropna(how='any', axis=0)
        fred_data = nrmlize(fred_data)

        uf, sf, vf = svd(fred_data,full_matrices=False)
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
        x_coef = np.zeros((s_num, H, X1.shape[1]+fred_data.shape[1]))
        pred2_t = np.zeros((s_num, H, X1.shape[1]+fred_data.shape[1])) # used to store the product of coefs and data
    else:
        x_coef = np.zeros((s_num, H, X1.shape[1]))
        pred2_t = np.zeros((s_num, H, X1.shape[1]))
    pred_t = pd.DataFrame(np.zeros((duration_len, s_num)), index=duration)  # 待將預測值放入的 DataFrame (可按時間做比較的結果)，之後也可以加入真實值做比較
    
    T_in_sample = np.zeros((s_num, H, 2), dtype=datetime)  # 待將每次迴歸 in-sample 時間的起始點與終點放入放入的空間 (dtype()：資料型態 要改成可存放時間的 datetime)
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
            reg = LassoCV(cv=cv, alphas=np.linspace(1.5, 0.001, 100), fit_intercept=True, max_iter=7000, random_state=2023).fit(X3, np.array(y).flatten())  # max_iter 由1200改成7000 才不會有"ConvergenceWarning"

            # 儲存迴歸結果
            t_b = t_a + rd(months=h)  # out-of-sample 的每個時間點
            coef[tid, h - 1, :] = reg.coef_[:f.shape[1]]
            if ('fred_data' in kwargs) == True:
                x_coef[tid, h - 1, :] = bd(np.sqrt(len(ux))*vx[:10, :].T,np.sqrt(len(uf))*vf[:10, :].T) @ bd(inv(np.diag(sx[:10])),inv(np.diag(sf[:10]))) @ reg.coef_[:f.shape[1]]
                pred2_t[tid, h-1, :] = np.c_[X1[t_a:t_a], fred_data[t_a:t_a]].flatten()* x_coef[tid, h-1, :] #+ dummy[t_b:t_b].values@reg.coef_[f.shape[1]:] + reg.intercept_
            else:
                x_coef[tid, h - 1, :] = np.sqrt(len(ux)) * vx[:10, :].T @ inv(np.diag(sx[:10])) @ reg.coef_[:f.shape[1]]
                pred2_t[tid, h-1, :] = X1[t_a:t_a].values.flatten()* x_coef[tid, h-1, :] #+ dummy[t_b:t_b].values@reg.coef_[f.shape[1]:] + reg.intercept_
            pred_t[tid][t_b:t_b] = np.c_[f[t_a:t_a], dummy[t_b:t_b]] @ reg.coef_ + reg.intercept_  # all the input array dimensions for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size 1 and the array at index 1 has size 0
            T_in_sample[tid, h - 1, 0] = rawYX.index[0]
            T_in_sample[tid, h - 1, 1] = rawYX.index[-1]

        if state == 'level':
            pred_t[tid][:t_a] = Y[:t_a]  # (都尚未是成長率)將真實值合入預測值
            pred_t[tid] = pred_t[tid][:t_d].pct_change(12, fill_method=None)  # 控制於'out-of-sample 的最後時間點'，否則會有多H期沒意義的結果，並影響mse估計
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




# -----------------------------------------------------------------------------


def out_mse(yf,yf_state,yr,t_process):
    """
    This "function" is mainly used to calculate the out-of-sample mean squared error (OOS MSE)
    
    Parameters
    ----------    
    yf : the prediction result
    yf_state : What is the state of "yf". Is the'growth rate' or 'level'.
    yr : truth value of the "level" state
    t_process : the last time point of "in-sample"
    
    Returns
    -------
    pred_gr : Contains two columns of DF. The result of merging the true value into the predicted value, and the result of purely only the true value
    out_mse : out-of-sample mean squared error (OOS MSE) of growth rate
    
    """

    if yf_state == 'level':
        yf[0][:t_process]=yr[:t_process] #(都尚未是成長率)將真實值合入預測值
        yf = yf.pct_change(12,fill_method=None) 
        col_name = 'forecast(level)'
        
    yr = yr.pct_change(12,fill_method=None)
    
    if yf_state == 'growth rate':
        yf[0][:t_process]=yr[:t_process] #(都已是成長率)將真實值合入預測值
        col_name = 'forecast'

    pred_gr = pd.concat([yf,yr],axis=1) #水平合併
    pred_gr.columns=[col_name,'real'] 
    t_a = t_process+rd(months=1)
    out_mse = np.mean((pred_gr['real'][t_a:]-pred_gr[col_name][t_a:]) ** 2) 
    
    return [pred_gr,out_mse]


class Get_Forecast:
    def __init__(self,key,X,fred_data,t_process_f,t_process_f_with_fred,H):  # constructor 建構子
        # store the whole information of forecasts
        self.X = X
        self.fred_data = fred_data
        self.result = {}
        self.result['forecast_without_fred_f'] = [forecast(key, 'level', X, t_process_f, t_process_f, H),
             forecast(key, 'growth rate', X, t_process_f, t_process_f, H)]
        self.result['forecast_with_fred_f'] = [forecast(key, 'level', X, t_process_f_with_fred, t_process_f_with_fred, H, fred_data=fred_data),
             forecast(key, 'growth rate', X, t_process_f_with_fred, t_process_f_with_fred, H, fred_data=fred_data)]
        # the forecasts
        self.forecasts = {}
        self.forecasts['forecast_without_fred_f'] = pd.concat(
            [self.result['forecast_without_fred_f'][0][3].iloc[:, 0].to_frame(),
             self.result['forecast_without_fred_f'][1][3].iloc[:, :]], axis=1)
        self.forecasts['forecast_with_fred_f'] = pd.concat(
            [self.result['forecast_with_fred_f'][0][3].iloc[:, 0].to_frame(),
             self.result['forecast_with_fred_f'][1][3].iloc[:, :]], axis=1)
        self.x_coefs = {}
        self.x_coefs['forecast_without_fred_f'] = [self.result['forecast_without_fred_f'][0][5], self.result['forecast_without_fred_f'][1][5]]
        self.x_coefs['forecast_with_fred_f'] = [self.result['forecast_with_fred_f'][0][5], self.result['forecast_with_fred_f'][1][5]]
        self.t_process_f = t_process_f
        self.t_process_f_with_fred = t_process_f_with_fred
        self.x_cb = {}
        self.x_cb['forecast_without_fred_f'] = [self.result['forecast_without_fred_f'][0][6], self.result['forecast_without_fred_f'][1][6]]
        self.x_cb['forecast_with_fred_f'] = [self.result['forecast_with_fred_f'][0][6], self.result['forecast_with_fred_f'][1][6]]
        self.H = H

    def Fig_With_RangeSlider(self,title):
        # 要繪製的時間
        #start_t = self.result['forecast_without_fred_f']['real'].index[0] + rd(months=self.H)
        self.forecasts['forecast_without_fred_f'] = self.forecasts['forecast_without_fred_f'].round(3)
        self.forecasts['forecast_with_fred_f'] = self.forecasts['forecast_with_fred_f'].round(3)
        # 要繪製的內容
        result_0 = self.forecasts['forecast_without_fred_f']['real'].dropna(how='any',axis=0)
        result_1 = self.forecasts['forecast_without_fred_f']['forecast(level)_0'][self.t_process_f:]
        result_2 = self.forecasts['forecast_without_fred_f']['forecast_0'][self.t_process_f:]
        result_3 = self.forecasts['forecast_with_fred_f']['forecast(level)_0'][self.t_process_f_with_fred:]
        result_4 = self.forecasts['forecast_with_fred_f']['forecast_0'][self.t_process_f_with_fred:]
        # Create figure
        fig = go.Figure()
        for data, Y_name,color  in zip([result_0, result_1, result_2, result_3, result_4],
                               ['real', 'forecast(level)', 'forecast', 'forecast(level) with fred data', 'forecast with fred data'],
                                 ['dimgray', 'dodgerblue','crimson','royalblue','indianred' ]):
            fig.add_trace(go.Scatter(x=data.index, y=data, name=Y_name, line=dict(color=color, width=2)))
        # Set title
        fig.update_layout(
            title={
                'text': title,
                'y': 0.95,  # 位置
                'x': 0.5,
                'xanchor': 'center',  # 相對位置
                'yanchor': 'top'}

             ,legend=dict(orientation="h",  # 水平顯示
                        y=1.03,x=1,
                        xanchor="right",#設置圖例的水平位置(極限點)
                        yanchor="top" #設置圖例的垂直位置(極限點)
                         )
            ,autosize=True
            ,height=600 #設置繪圖的高度（以像素為單位）

            ,margin = dict(b=30, #設置下邊距（以像素為單位）
                          t=110, #設置上邊距（以像素為單位）
                          l=30, #設置左邊距（以像素為單位）
                          r=30, #設置右邊距（以像素為單位）
            )
            #,xaxis_title = 'Dates' #橫軸軸標籤
            , font=dict(size=16) #設置字體大小
            , template='plotly_white' #換默認模板
            # "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"
            , hovermode="x"
        )
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, #設置範圍要採取的步數
                             label="1m",
                             step="month", #count值設置範圍所依據的度量單位
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD", #Data for year to date etc.
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date",
            )
        )
        return fig

    def figu_heatmap(coef, figu_name, vmin, vmax):
        # This code is used for producing the time-varying coefficients based on different training windows
        # it does not work so far! 2023/5/16
        fig1, axes = plt.subplots(4, 3, figsize=(36, 24))  # np.size(coef,1)=12，4*3=12；figsize 用來設置圖形大小

        for h in range(np.size(coef, 1)):
            sns.heatmap(coef[:, h, :].T, ax=axes.flat[h], linewidths=0.05, linecolor="grey", cmap='seismic', vmin=vmin,
                        vmax=vmax)

            axes.flat[h].set_xlabel('In-Sample_i', fontsize=12)
            # 先要有 ticks ， ticklabels 才能有效發會作用
            axes.flat[h].set_xticks(np.linspace(0, np.size(coef, 0), 12))  # "1/2" 僅是為了讓之後的 xticklabels 能在色塊區間置中
            axes.flat[h].set_xticklabels(np.linspace(1, np.size(coef, 0), 12).astype(int),
                                         rotation=45)  # astype(int) 留整數，rotation 旋轉

            axes.flat[h].set_ylabel('Beta_i', fontsize=12, rotation=360, labelpad=13)
            axes.flat[h].set_yticks(
                np.arange(1 / 2, np.size(coef, 2) + 1 / 2, 1))  # "1/2" 僅是為了讓之後的 yticklabels 能在色塊區間置中
            axes.flat[h].set_yticklabels(np.arange(1, np.size(coef, 2) + 1, 1), rotation=360)

            h1 = h + 1
            axes.flat[h].set_title('Forecast  t + %s  period' % h1, fontsize=14)  # %後面不能直接放h+1，所以先建h1

        plt.subplots_adjust(wspace=0, hspace=0.4)
        plt.suptitle(figu_name, fontsize=26)
        plt.show()

    def figu_heatmap2(self, name, state, figu_name, vmin, vmax, figsize, data_type):
        if data_type == 'coef':
            # This code is used for producing the time-varying coefficients based on different h's
            fig1, axes = plt.subplots(1, 1, figsize=figsize)  # np.size(coef,1)=12，4*3=12；figsize 用來設置圖形大小
            idx2 = 1
            if state == 'level':
                idx2 = 0
            coef = self.x_coefs[name][idx2][0, :, :]
            if name == 'forecast_without_fred_f':
                coef = pd.DataFrame(coef, columns=self.X.columns)
            else:
                coef = pd.DataFrame(coef, columns=list(self.X.columns) + list(self.fred_data.columns))
            sns.heatmap(coef, linewidths=0.05, linecolor="grey", cmap='seismic', vmin=vmin,
                        vmax=vmax)
        elif data_type == 'contribution':
            fig1, axes = plt.subplots(1, 1, figsize=figsize)  # np.size(coef,1)=12，4*3=12；figsize 用來設置圖形大小
            idx2 = 1
            if state == 'level':
                idx2 = 0
            cb = self.x_cb[name][idx2][0, :, :]
            if name == 'forecast_without_fred_f':
                cb = pd.DataFrame(np.abs(cb), columns=self.X.columns)
            else:
                cb = pd.DataFrame(np.abs(cb), columns=list(self.X.columns) + list(self.fred_data.columns))
            sns.heatmap(cb, linewidths=0.05, linecolor="grey", cmap='seismic', vmin=vmin,
                        vmax=vmax)
        axes.invert_yaxis()
        #axes.set_title('Forecast  t + %s  period' % h1, fontsize=14)  # %後面不能直接放h+1，所以先建h1

        plt.subplots_adjust(wspace=0, hspace=0.4)
        #plt.set_title(figu_name, fontsize=26)
        plt.show()

    def to_excel(self, name, state, result_path, file_name):
        """
        #透過 file_name 區分是進口還是出口的係數資料
        #舉例設定:file_name = 'im_coeff_heatmap' if model==im_model else 'ex_coeff_heatmap'
        """
        idx2 = 1
        if state == 'level':
            idx2 = 0
        coef = self.x_coefs[name][idx2][0, :, :]
        if name == 'forecast_without_fred_f':
            coef = pd.DataFrame(coef, columns=self.X.columns)
        else:
            coef = pd.DataFrame(coef, columns=list(self.X.columns) + list(self.fred_data.columns))
        sheet_name = name + '_' + state
        try:  # 判斷檔案是否存在(若已存在就用舊檔更新，若不存在則建新檔案，否則永遠只會覆蓋檔案，並指有一個sheet)
            with pd.ExcelWriter(result_path + '\\' + file_name + '.xlsx', engine='openpyxl',
                                mode='a') as writer:  # mode='a'現有檔案讀寫
                try:  # 判斷舊檔案中，是否有相同sheet_name，有的話要先刪除，否則會有多餘的
                    # 先刪除原來的sheet
                    book = writer.book
                    book.remove(book[sheet_name])  #
                except KeyError:
                    pass
                # 存到指定的sheet
                coef.to_excel(writer, sheet_name=sheet_name)
                print('讀取舊檔案',file_name,'，更新sheet_name=', sheet_name, '之內容')

        except FileNotFoundError:
            writer = pd.ExcelWriter(result_path + '\\' + file_name + '.xlsx',
                                    engine='openpyxl')  # 設定路徑及檔名，並指定引擎openpyxl

            coef.to_excel(writer, sheet_name=sheet_name)
            writer.save()  # 存檔生成excel檔案
            print('不存在舊檔案，開啟新檔案',file_name,'，加入sheet_name=', sheet_name, '之內容')

    """    def to_excel(self, name, state, result_path):
            writer = pd.ExcelWriter(result_path + '/coeff_heatmap.xlsx', engine='openpyxl')  # 設定路徑及檔名，並指定引擎openpyxl
            idx2 = 1
            if state == 'level':
                idx2 = 0
            coef = self.x_coefs[name][idx2][0, :, :]
            if name == 'forecast_without_fred_f':
                coef = pd.DataFrame(coef, columns=self.X.columns)
            else:
                coef = pd.DataFrame(coef, columns=list(self.X.columns) + list(self.fred_data.columns))
            coef.to_excel(writer, sheet_name=name+'_'+state)
            writer.save()  # 存檔生成excel檔案
    """