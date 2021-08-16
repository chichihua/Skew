# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:41:36 2020

Delta Curve Skew

@author: AXZQ
"""


### Delta Curve Call Skew： 0.25 C - 0.5 C

from datetime import datetime, timedelta, date
from dateutil.parser import parse
import numpy as np
import pandas as pd
from WindPy import w
import matplotlib.pyplot as plt
import py_vollib.black.implied_volatility as biv
import py_vollib.black.greeks.analytical as bga
import statsmodels.api as sm
from scipy.optimize import curve_fit


w.start()


######期货标的

product = 'I'
exchange = 'DCE'
code = '.DCE'


"""
product = 'M'
exchange = 'DCE'
code = '.DCE'
"""
"""
product = 'CU'
exchange = 'SHFE'
code = '.SHF'
"""
####算skew的时间区间
start =  "2020-01-01"
end = "2020-12-30"
N = 5  # 拟合选取的最少合约数,159合约 N = 5,连月合约 N = 3


###参数
t1 =-15 ###到期前t1天换主力月合约,159合约 t1 = -10,连月合约 t1 = -6


def form(date):
    return parse(str(date)).strftime('%Y/%m/%d')


def my_implied_vol(S, F, K, r, T, option_type):
    try:
        my_impVol = biv.implied_volatility_of_discounted_option_price(S, F, K, r,T, option_type.lower())
    except:
        my_impVol = np.nan
    return my_impVol


def calc_basic_greeks(df,F,T):
    if df.empty:
        df = df.reindex(columns=df.columns.tolist() + ['delta', 'gamma', 'vega', 'theta'])
        return df
    df['DELTA'] = df.apply(lambda row: bga.delta(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility']), axis=1)
    df['GAMMA'] = df.apply(lambda row: bga.gamma(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility']), axis=1)
    df['VEGA'] = df.apply(lambda row: bga.vega(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility']), axis=1)
    df['THETA'] = df.apply(lambda row: bga.theta(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility'])*365/245, axis=1)
    return df


def Cal_greeks(day,options):
    df = options.dropna()
    df['EXE_ENDDATE'] = df['EXE_ENDDATE'].apply(lambda x: datetime.strftime(datetime.strptime(str(x), "%Y/%m/%d"), "%Y%m%d"))
    F = df['F'][0]
    T = w.tdayscount(day, df['EXE_ENDDATE'].iloc[0], "").Data[0][0]/245
    r = 0
    df = df.dropna()
   # strike = df['Strike'].loc[df['Close_price'].isna()]
   # for i in strike:
   #     df = df[df["Strike"] != i]
   # c = np.unique(df[df.Type == 'c']['Strike'])
   # p = np.unique(df[df.Type == 'p']['Strike'])
   # strike = list(set(c).intersection(p))
   # temp = pd.DataFrame([])
   # for i in strike:
   #     temp = pd.concat([df[df["Strike"] == i],temp],axis=0)
   # df = temp
    
    #计算隐含波动率
    df['Implied Volatility'] = df.apply(lambda row: my_implied_vol(row['Close_price'], F, row['Strike'], r,T, row['Type']), axis=1)
    df.loc[df['Implied Volatility'] < 0, 'Implied Volatility'] = np.nan
    pd.options.mode.chained_assignment = None  # default='warn'  
    df = df.dropna()
    #strike = df['Strike'].loc[df['Implied Volatility'].isna()]
    #for i in strike:
    #   df = df[df["Strike"] != i]
       
    # strike > F 的合约
    otmc = df[(df['Type'] == 'c') & (df.Strike > F)]
    otmc = calc_basic_greeks(otmc,F,T) 
   
    #strike < F 的合约
    otmp = df[(df['Type'] == 'p') & (df.Strike <= F)]
    otmp = calc_basic_greeks(otmp,F,T)
        
    df = otmc.append(otmp, sort=False)[otmc.columns.tolist()].sort_values(['Strike'])
    df.insert(0, 'Contract', list(df.index))
    df.index = [i for i in range(len(df))]
    df['T'] = T
    return df



def iv_parameterization(otm_data, weighting=0):
    group = otm_data.dropna()  
    F = group['F'].iloc[0]
    term = group['T'].iloc[0]
    group = group.sort_values(by='Strike')
    group['moneyness'] = group['Strike'].apply(lambda x: x / F - 1)
    group['xk'] = group['Strike'].apply(lambda x: np.log(x / F)) # log moneyness
    group['yk'] = group['Implied Volatility'] ** 2 * term
    X = pd.DataFrame(sm.add_constant(group['xk'].values), index=group.index)
    X.columns = ['const', 'log_moneyness']
    X['squared_log_moneyness'] = X['log_moneyness'] ** 2
    Y = group['yk']
    group['weights'] = group['VEGA'] 

    if weighting and not group['weights'].isna().all():
        weighted_X = pd.DataFrame(columns=X.columns)
        weighted_Y = pd.DataFrame()
        for column in X.columns:
            weighted_X[column] = X[column] * group['weights']
        weighted_Y = group['yk'] * group['weights']
        model = sm.OLS(weighted_Y, weighted_X)
    else:
        model = sm.OLS(Y, X)

    result = model.fit()
    params = np.array(result.params).reshape(3, 1)
    group['parameterized_yk'] = np.dot(X, params)
    group['parameterized_IV'] = np.sqrt(group['parameterized_yk'] / term)
    group = group[['Implied Volatility', 'DELTA','parameterized_IV']]
    return group


def delta_surface_building(group):
    func = (lambda x, a, b, c: a * x ** 2 + b * x + c)
    deltas = [0.05 * i for i in range(2, 19)]
    columns = deltas
    curves = pd.DataFrame(columns=columns)
    group.drop(group[(group['DELTA'] < 0.1) | (group['DELTA'] > 0.9)].index, inplace=True)
    group.dropna(inplace=True)
    if group.shape[0] >= N:
        X = np.array(group['DELTA'])
        Y = np.array(group['parameterized_IV'])
        popt, pcov = curve_fit(func, X, Y, bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
        curves = list(func(np.array(deltas), *popt))
    else: #return NA
        curves = list(np.array(deltas) * np.nan)         
    return curves #ATM_IV里仅包含当期合约个数小于5的到期日和ATM IV，合约数>=5的不包含在内

def calc_vol_surface(df):
    otm_p = df[df.Type == 'p']
    if len(otm_p)!=0:
        otm_p['DELTA'] = 1 - abs(otm_p['DELTA']) 
    otm_c = df[df.Type == 'c']
    otm_data = otm_p.append(otm_c, sort=False)
    #step 1: raw IV w/ Strike
    group = iv_parameterization(otm_data, weighting=1)
    #step 2: delta surface smoothing IV w/ Delta
    curves = delta_surface_building(group)
    #curves.columns = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',\
                   #   '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9']
                   # 小于 0.5的 是 call iv， 大于0.5的是 put iv
    return curves

#期货数据
temp = w.wset("optionfuturescontractbasicinf","exchange="+exchange+";productcode="+product+";contract=total")
Futures = np.unique(temp.Data[2])
data = w.wsd(list(Futures), "close", start, end, "")
df_P = pd.DataFrame(data.Data).T
df_P.index = pd.DataFrame(data.Times)
df_P.columns = pd.DataFrame(data.Codes)
data = w.wsd(list(Futures), "volume", start, end, "")
df_V = pd.DataFrame(data.Data).T
df_V.index = pd.DataFrame(data.Times)
df_V.columns = pd.DataFrame(data.Codes)
df_V['max_idx'] = df_V.idxmax(axis=1)

#期权列表
data_option = w.wset("optionfuturescontractbasicinf", "exchange=" + exchange + ";productcode=" + product + ";contract=total;field=wind_code,\
option_mark_code,call_or_put,exercise_price,expire_date")
df_list = pd.DataFrame(data_option.Data).T
df_list.columns = ['Code','Futures','Type','Strike','EXE_ENDDATE']
df_list = df_list.replace('认沽', 'p')
df_list = df_list.replace('认购', 'c')
df_list['Code'] = df_list['Code'] + code
df_list['EXE_ENDDATE'] = df_list['EXE_ENDDATE'].apply(form)


##计算得出skew的时序
skew = []
index = []
for i in range(len(df_V)):
    Futures_code = df_V['max_idx'][i]
    Futures_price = df_P[Futures_code][i]
    day = datetime.strftime(df_V.index[i][0],"%Y-%m-%d")
    options = df_list[df_list['Futures'] == Futures_code[0]]
    expire_date = options['EXE_ENDDATE'].iloc[0]
    end_date = w.tdaysoffset(t1, expire_date, "").Data[0][0]
    end_date =  datetime.date(end_date)
    if df_V.index[i][0] < end_date:    
        contracts = list(options["Code"])
        options.index = options['Code']
        options = options[['Type','Strike','EXE_ENDDATE']]
        data_options = w.wsd(contracts,"close",day,day,"")
        temp = pd.DataFrame(data_options.Data).T
        temp.index = data_options.Codes
        temp.columns = ['price']
        options['Close_price'] = temp['price']
        options['F'] = Futures_price
        index.append(Futures_code[0])
    else:
        temp = df_V.drop(Futures_code,1)
        temp = temp.drop('max_idx',1)
        temp['secondmax_idx'] = temp.idxmax(axis=1)
        Futures_code_new = temp['secondmax_idx'][i]
        Futures_price_new = df_P[Futures_code_new][i]
        options = df_list[df_list['Futures'] == Futures_code_new[0]]
        contracts = list(options["Code"])
        options.index = options['Code']
        options = options[['Type','Strike','EXE_ENDDATE']]
        data_options = w.wsd(contracts,"close",day,day,"")
        temp = pd.DataFrame(data_options.Data).T
        temp.index = data_options.Codes
        temp.columns = ['price']
        options['Close_price'] = temp['price']
        options['F'] = Futures_price_new
        index.append(Futures_code_new[0])


######利用拟合计算skew ############       
    ####计算IV,GREEKS
    df = Cal_greeks(day,options)

    ####拟合IV曲线
    curve = calc_vol_surface(df)
    
    ###计算skew
    #skew.append((curve[13]-curve[3])/curve[8])
    skew.append((curve[3]-curve[8]))
    
###### 上述依旧在for循环里面######

    
skew = pd.DataFrame(skew)
skew.index = df_V.index
skew.columns=['skew']
skew['Futures_code'] = pd.DataFrame(index,index=df_V.index)
skew.to_excel('Cskew.xls')


#####回测参数
T1 = 60  #T1天前到现在的skew分位数
Low = 30  #分位数
High = 80 #分位数



####回测
data = pd.read_excel('D:\AX_Python\商品Skew回测\\Cskew.xls')  
       #运行时也要保证Python右上角打开的文件夹在此目录下
data.index = skew.index
data = data[['skew','Futures_code']]
P_low = []
P_median = []
P_high = []
flag = pd.DataFrame([],columns=['flag'])
for i in range(len(data)-T1-1):
    temp = data.iloc[i:i+T1,0].dropna()
    p1 = np.percentile(temp, Low)
    p2 = np.median(temp)
    p3 = np.percentile(temp, High)
    P_low.append(p1)
    P_median.append(p2)
    P_high.append(p3)
    day = data.index[i+T1][0]
    if data.iloc[i+T1,0]> p3:
        flag.loc[day] = 0
    elif data.iloc[i+T1,0]>= p2:
        flag.loc[day] = 1
    elif data.iloc[i+T1,0]>= p1:
        flag.loc[day] = 2
    else:
        flag.loc[day] = 3
        
flag['pos'] = 0
flag['futures'] = np.nan
flag['Long'] = np.nan
flag['Short'] = np.nan
flag['p&l_Long'] = np.nan   
flag['p&l_Short'] = np.nan 
flag['skew'] = np.nan
flag['30%'] = np.nan
flag['50%'] = np.nan
flag['80%'] = np.nan

flag.iloc[0,2] = data.iloc[T1,1]
flag.iloc[0,-4] = data.iloc[T1,0]
flag.iloc[0,-3] = P_low[0]
flag.iloc[0,-2] = P_median[0]
flag.iloc[0,-1] = P_high[0]
#flag.iloc[-1,2] = data.iloc[-2,1]
for i in range(1, len(flag)):
    flag.iloc[i,2] = data.iloc[i+T1,1]
    flag.iloc[i,-4] = data.iloc[i+T1,0]
    flag.iloc[i,-3] = P_low[i]
    flag.iloc[i,-2] = P_median[i]
    flag.iloc[i,-1] = P_high[i]
    
    if flag.iloc[i-1,1] == 0:
        if flag.iloc[i,0] == 0:
            flag.iloc[i,1] = 1
        elif flag.iloc[i,0] == 3:
            flag.iloc[i,1] = -1
        else:
            flag.iloc[i,1] = 0
    elif  flag.iloc[i-1,1] == 1:
        if flag.iloc[i,0] == 2:
            flag.iloc[i,1] = 0
        elif flag.iloc[i,0] == 3:
            flag.iloc[i,1] = -1
        else:
            flag.iloc[i,1] = 1
    else:
        if flag.iloc[i,0] == 1:
            flag.iloc[i,1] = 0
        elif flag.iloc[i,0] == 0:
            flag.iloc[i,1] = 1
        else:
            flag.iloc[i,1] = -1
            
    ####建仓
    if flag.iloc[i,1] == 1:
        flag.iloc[i,3] = df_P[(flag.iloc[i,2],)][i+T1]
    elif flag.iloc[i,1] == -1:
        flag.iloc[i,4] = df_P[(flag.iloc[i,2],)][i+T1]
    else:
        flag = flag

    ####平仓  
    if flag.iloc[i-1,1] == -1:
        if flag.iloc[i-1,1] != flag.iloc[i,1]:
            flag.iloc[i,4] = -df_P[(flag.iloc[i-1,2],)][i+T1]
        else:
            flag = flag
    elif flag.iloc[i-1,1] == 1:
        if flag.iloc[i-1,1] != flag.iloc[i,1]:
            flag.iloc[i,3] = -df_P[(flag.iloc[i-1,2],)][i+T1]
        else:
            flag = flag
    else:
        flag=flag


start1 = 0
start2 = 0
for i in range(1,len(flag)):
    if flag.iloc[i,3] != np.nan and  np.isnan(flag.iloc[i-1,3]):
        start1 = flag.iloc[i,3]
    elif np.isnan(flag.iloc[i,3]) and  flag.iloc[i-1,3] < 0:
        flag.iloc[i-1,5] = (-flag.iloc[i-1,3]-start1)/start1
        start1 = 0
    elif np.isnan(flag.iloc[i,3]) and  flag.iloc[i-1,3] >= 0:
        print("wrong data")
    elif flag.iloc[i,3] >0 and flag.iloc[i-1,3] >0:
        if flag.iloc[i,2] != flag.iloc[i-1,2]:
            flag.iloc[i,5] = (df_P[(flag.iloc[i-1,2],)][i+T1]-start1)/start1
            start1 = flag.iloc[i,3]
        else:
            flag = flag
    else:
        flag = flag
        start1 = start1
        
    if flag.iloc[i,4] != np.nan and  np.isnan(flag.iloc[i-1,4]):
        start2 = flag.iloc[i,4]
    elif np.isnan(flag.iloc[i,4]) and  flag.iloc[i-1,4] < 0:
        flag.iloc[i-1,6] = (-flag.iloc[i-1,4]-start2)/start2
        start2 = 0
    elif np.isnan(flag.iloc[i,4]) and  flag.iloc[i-1,4] >= 0:
        print("wrong data")
    elif flag.iloc[i,4] > 0 and flag.iloc[i-1,4] > 0:
        if flag.iloc[i,2] != flag.iloc[i-1,2]:
            flag.iloc[i,6] = (df_P[(flag.iloc[i-1,2],)][i+T1]-start2)/start2
            start2 = flag.iloc[i,4]
        else:
            flag = flag
    else:
        flag = flag
        start2 = start2
 

flag.to_excel('result.xls')       
Strategy1 = flag['p&l_Long'].dropna()
Strategy2 = flag['p&l_Short'].dropna()
S1_pl = np.mean(Strategy1)
S2_pl = np.mean(Strategy2)
S1_pro = sum(Strategy1>0)/len(Strategy1)
S2_pro = sum(Strategy2>0)/len(Strategy2)


######画图

plt.plot(flag['skew'],label='skew')
plt.plot(flag['30%'],label='30%')
plt.plot(flag['50%'],label='50%')
plt.plot(flag['80%'],label='80%')

import pylab as pl
pl.xticks(rotation=60)
plt.legend(loc='best')
plt.show()
    

print("---------SPLC---------")       
print(Strategy1)

print("---------LPSC---------")   
print(Strategy2)
        
        