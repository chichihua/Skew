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

# =============================================================================
# # 需要drop掉带A合约！！！！！！！！！！！待
# =============================================================================


####算skew的时间区间
# start =  "2021-08-10"
# end = "2021-08-16"

start =  "2019-02-25"
end = "2019-02-25"
# start =  "2020-10-01"
# end = "2020-10-30"
Spot = "510050.SH"
code = '.SH'
N = 3  # 拟合选取的最少合约数,159合约 N = 5,连月合约 N = 3

###参数
t1 =5###到期前t1天换主力月合约,159合约 t1 = -10,连月合约 t1 = -6

######读取Opt_ID_Excel
# RData= pd.read_excel('D:\安信工作文件\Excel_Greeks监控\ETF_Skew\Skew//50Opt_ID.xlsx')  # today成交流水
RData= pd.read_excel('./data/50Opt_ID.xlsx')  # today成交流水

OptID = RData.drop(0)

OptID.rename(columns={'wind_code':'ID','sec_name':'name','option_mark_code':'spot', 'call_or_put':'CP',\
                      'exercise_price':'K','contract_unit':'Mul','limit_month':'month',\
                          'listed_date':'SDate','exercise_date':'EDate','contract_state':'state'}, inplace=True)

#######################
r_bs=0.03
r_black=0

method = 'black'
#######################


if method == 'bs':
    r = r_bs
elif method == 'black':
    r = r_black


######定义fuction



def form(date):
    return parse(str(date)).strftime('%Y/%m/%d')


def my_implied_vol(Pri, F, K, r, T, option_type):
    try:
        my_impVol = biv.implied_volatility_of_discounted_option_price(Pri, F, K, r,T, option_type.lower())
    except:
        my_impVol = np.nan
    return my_impVol



def calc_f(call_prices, put_prices, call_ids, atm_call, k_list, r, t):
   '''
   计算同一到期日ATM以及上、下2档的期权的合成期货价格，取均值求得forward
   若期权价格出现nan值，则取最近一档替代，assert至少有三个行权价的合成期货价格
   :param call_prices: call option 价格list, 用于计算合成期货价格
   :param put_prices: put option 价格list, 用于计算合成期货价格
   :param call_ids: call option id list, 用于定位档位
   :param atm_call: atm call id, 用于定位档位
   :param k_list: strike list, 用于计算合成期货价格
   :param r: risk-free rate
   :param t: calc_t
   :return: 合成期货均价forward
   '''
   total = 0
   count = 0
   synfutures_n = 2 # n=2，用于下方cap = 5
   
   cap = synfutures_n * 2 + 1
   loc = np.where(np.array(call_ids) == atm_call)[0][0]
   #loc = 9
      
   for n in range(len(call_prices)):
       if n == 0:
           if np.isnan(call_prices[loc]) == False \
                   and np.isnan(put_prices[loc]) == False \
                   and np.isnan(k_list[loc]) == False:
               # S = C + K*e^{-r*t} - P
               total += call_prices[loc] - put_prices[loc] + k_list[loc] * np.exp(- r * t)
               count += 1
               # count += 1表示count = count+1
               if count == cap:
                   break
           else:
               cap = synfutures_n * 2  # 若ATM价格为nan，取两边，##合成期货时使用上下2档，f function里面cap就是capacity？？？
       else:
           try:
               if loc - n >= 0:
                   if np.isnan(call_prices[loc+n]) == False \
                           and np.isnan(put_prices[loc+n]) == False \
                           and np.isnan(k_list[loc+n]) == False \
                           and np.isnan(call_prices[loc-n]) == False \
                           and np.isnan(put_prices[loc-n]) == False \
                           and np.isnan(k_list[loc-n]) == False:
                       total += call_prices[loc+n] - put_prices[loc+n] + k_list[loc+n] * np.exp(- r * t) \
                                + call_prices[loc-n] - put_prices[loc-n] + k_list[loc-n] * np.exp(- r * t)
                                # 同时计算 ATM+1 & ATM-1的syn，因此下方count +=2
                       count += 2
                       if count == cap:
                           break
           except:
               pass

   # assert count >= 3 #若 >=3，在边界处会报错
   assert count >= 1
   f = total / count
   return f

#syn_f = calc_f(call_prices, put_prices, call_ids, atm_call, k_list, r, t)


def calc_basic_greeks(df,F,T):
    if df.empty:
        df = df.reindex(columns=df.columns.tolist() + ['delta', 'gamma', 'vega', 'theta'])
        return df
    # def创建的方法是有名称的，而lambda没有。apply(func,*args,**kwargs),axis=1表示按照行进行操作。
    df['DELTA'] = df.apply(lambda row: bga.delta(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility']), axis=1)
    df['GAMMA'] = df.apply(lambda row: bga.gamma(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility']), axis=1)
    df['VEGA'] = df.apply(lambda row: bga.vega(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility']), axis=1)
    df['THETA'] = df.apply(lambda row: bga.theta(row['Type'], F, row['Strike'], T, 0, row['Implied Volatility'])*365/245, axis=1)
    return df


def Cal_greeks(date,Options_m0):
    df = Options_m0.dropna()
    df['EndDate'] = df['EndDate'].apply(lambda x: datetime.strftime(datetime.strptime(str(x), "%Y/%m/%d"), "%Y%m%d"))
    F = syn_f
    # T = w.tdayscount(date, df['EndDate'].iloc[0], "").Data[0][0]/245
    # r = 0
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
    df['Implied Volatility'] = df.apply(lambda row: my_implied_vol(row['Close_price'], F, row['Strike'], r,t, row['Type']), axis=1)
    df.loc[df['Implied Volatility'] < 0, 'Implied Volatility'] = np.nan
    pd.options.mode.chained_assignment = None  # default='warn'，忽略警告。
    df = df.dropna()
    #strike = df['Strike'].loc[df['Implied Volatility'].isna()]
    #for i in strike:
    #   df = df[df["Strike"] != i]
       
    # strike > F 的合约
    otmc = df[(df['Type'] == 'c') & (df.Strike > F)]
    otmc = calc_basic_greeks(otmc,F,t) 
   
    #strike < F 的合约
    otmp = df[(df['Type'] == 'p') & (df.Strike <= F)]
    otmp = calc_basic_greeks(otmp,F,t)
    
    # df = otmc.append(otmp, sort=False)[otmc.columns.tolist()].sort_values(['Strike'])
    # 上面方法对于OTMC = empty 会造成 df的greeks是nan,如2019/2/25
    if len(otmc) == 0:
        df = otmp
    elif len(otmp) == 0:
        df = otmc
    else:
        df = otmc.append(otmp, sort=False)[otmc.columns.tolist()].sort_values(['Strike'])
    
    #把otmp加到otmc下方，sort_values()函数用于order by
    df.insert(0, 'Contract', list(df.index)) #把index转变成list
    df.index = [i for i in range(len(df))] # renew index
    df['T'] = t
    return df



def iv_parameterization(otm_data, weighting=0):
    group = otm_data.dropna()  
    # F = group['F'].iloc[0]
    F = syn_f
    term = group['T'].iloc[0]
    group = group.sort_values(by='Strike')
    group['moneyness'] = group['Strike'].apply(lambda x: x / F - 1)
    # strike 转变为moneyness
    group['xk'] = group['Strike'].apply(lambda x: np.log(x / F)) # log moneyness
    group['yk'] = group['Implied Volatility'] ** 2 * term 
    # iv^2 * t
    X = pd.DataFrame(sm.add_constant(group['xk'].values), index=group.index)
    # sm.add_constant,为模型增加常数项，即回归线在 y 轴上的截距
    X.columns = ['const', 'log_moneyness']
    X['squared_log_moneyness'] = X['log_moneyness'] ** 2
    Y = group['yk']
    group['weights'] = group['VEGA'] 

    weighting = 0

    if weighting and not group['weights'].isna().all():
        weighted_X = pd.DataFrame(columns=X.columns)
        weighted_Y = pd.DataFrame()
        for column in X.columns:
            weighted_X[column] = X[column] * group['weights']
        weighted_Y = group['yk'] * group['weights']
        model = sm.OLS(weighted_Y, weighted_X)
    else:
        model = sm.OLS(Y, X) #最小二乘法

    result = model.fit()
    params = np.array(result.params).reshape(3, 1)
    group['parameterized_yk'] = np.dot(X, params)
    group['parameterized_IV'] = np.sqrt(group['parameterized_yk'] / term)
    group = group[['Implied Volatility', 'DELTA','parameterized_IV']]
    return group


def delta_surface_building(group):
    func = (lambda x, a, b, c: a * x ** 2 + b * x + c)
    # func = (lambda x, a, b, c, d: a * x ** 3 + b * x ** 2 + c * x + d) 
    #用3次函数拟合,可能过度拟合，详见Qwin_skew excel
    deltas = [0.05 * i for i in range(2, 19)]
    columns = deltas
    curves = pd.DataFrame(columns=columns)
    group.drop(group[(group['DELTA'] < 0.1) | (group['DELTA'] > 0.9)].index, inplace=True)
    # group.drop(group[(group['DELTA'] < 0.05) | (group['DELTA'] > 0.95)].index, inplace=True)
    group.dropna(inplace=True)
    if group.shape[0] >= N:
        X = np.array(group['DELTA'])
        Y = np.array(group['parameterized_IV'])
        popt, pcov = curve_fit(func, X, Y, bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
        # popt, pcov = curve_fit(func, X, Y, bounds=([0, -np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf]))
        curves = list(func(np.array(deltas), *popt))
        # print('')
    else: #return NA
        curves = list(np.array(deltas) * np.nan)         
    return curves #ATM_IV里仅包含当期合约个数小于5的到期日和ATM IV，合约数>=5的不包含在内

def calc_vol_surface(df):
    otm_p = df[df.Type == 'p']
    if len(otm_p)!=0:
        otm_p['DELTA'] = 1 - abs(otm_p['DELTA'])  # 用otm_p的iv
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


#Spot数据
"""
#temp = w.wset("optionfuturescontractbasicinf","exchange="+exchange+";productcode="+product+";contract=total")
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

"""

data = w.wsd(Spot, "close", start, end, "") 
df_P = pd.DataFrame(data.Data).T #把wsd数据转化成dataframe
df_P.index = pd.DataFrame(data.Times) #df_P添加日期为index
df_P = df_P.reset_index()
df_P.columns = ['Date','Price']




#期权列表(需要修改,不用Wset，直接读取Opt_ID的excel)
"""
data_option = w.wset("optionfuturescontractbasicinf", "exchange=" + exchange + ";productcode=" + product + ";contract=total;field=wind_code,\
option_mark_code,call_or_put,exercise_price,expire_date")
# 不能调用Wset，wind数据会超限。
"""
data_Option = OptID[['ID','name','spot','CP','K','SDate','expire_date']]
data_Option.columns = ['ID','name','spot','Type','Strike','StaDate','EndDate'] # columns index重命名
data_Option = data_Option.replace('认沽', 'p') # replace dataframe 里面的 值
data_Option = data_Option.replace('认购', 'c')
data_Option['ID'] = data_Option['ID'] + code
data_Option['EndDate'] = data_Option['EndDate'].apply(form)
data_Option['StaDate'] = data_Option['StaDate'].apply(form)
                # DataFrame.apply(func,....),将所有结果组合成一个Series数据结构并返回。

                

"""
product = 'M'
exchange = 'DCE'
code = '.DCE'
"""
# =============================================================================
# df_list = pd.DataFrame(data_option.Data).T
# df_list.columns = ['Code','Futures','Type','Strike','EXE_ENDDATE']
# df_list = df_list.replace('认沽', 'p')
# df_list = df_list.replace('认购', 'c')
# df_list['Code'] = df_list['Code'] + code
# df_list['EXE_ENDDATE'] = df_list['EXE_ENDDATE'].apply(form)
# =============================================================================
           


##计算得出skew的时序
skew = []
index = []
for i in range(len(df_P)):
    #Futures_code = df_V['max_idx'][i]
    #Futures_price = df_P[Futures_code][i]
    Spot_price = df_P['Price'][i]  
    #day = datetime.strftime(df_V.index[i][0],"%Y-%m-%d")
    date = datetime.strftime(df_P['Date'][i][0],"%Y-%m-%d")
                        # df_P['Date'][i]是tuple,df_P['Price'][i]是float
    #options = df_list[df_list['Futures'] == Futures_code[0]]
    #expire_date = options['EXE_ENDDATE'].iloc[0]
    # end_date = w.tdaysoffset(t1, expire_date, "").Data[0][0]
    # end_date =  datetime.date(end_date)
    lastdate = w.tdaysoffset(t1, date, "").Data[0][0] #往后倒推t1交易日,到期日需要大于lastdate
    lastdate = datetime.date(lastdate) # datetime.date    
    #lastdate = datetime.strftime(lastdate,"%Y-%m-%d") # str
            # 将date转化为和date一致的str
            
  ############## 转化成datetime才能date比较大小######################          
    SD = pd.to_datetime(data_Option['StaDate'], format="%Y-%m-%d")
    pddate = pd.to_datetime(date, format="%Y-%m-%d")
    ED = pd.to_datetime(data_Option['EndDate'], format="%Y-%m-%d")
    lastdate = pd.to_datetime(lastdate, format="%Y-%m-%d")
    
    options =data_Option[(SD < pddate) \
                & (ED > lastdate) ] ##选出此时存续的合约
    options = options[options['name'].str.contains('A') == False] #剔除带A的合约
    
    Options_m0 = options[options['EndDate'] == min(options['EndDate'])]  ##选出当月合约
    exp_date = pd.to_datetime(Options_m0['EndDate'], format="%Y-%m-%d")
    exp_date = pd.to_datetime(str(exp_date.values[0])) # values用于提取series数据
    
    end_date = w.tdaysoffset(-t1, form(exp_date), "").Data[0][0] #往前倒推t1交易日
    end_date = pd.to_datetime(end_date, format="%Y-%m-%d") #转化形式用于下方 if compare！
  ##################################################################  
    
    
    # if df_P['Date'][i][0] < end_date:    
    contracts = list(Options_m0["ID"])
    Options_m0.index = Options_m0['ID']
    Options_m0 =  Options_m0[['name','Type','Strike','EndDate']]
    #data_options = w.wsd(contracts,"close",date,date,"")
    options_close = w.wsd(contracts,"close",date,date,"")
    temp = pd.DataFrame(options_close.Data).T # 把wind的data数据转化成dataframe形式
    temp.index = options_close.Codes # index为wind code
    temp.columns = ['price']
    Options_m0['Close_price'] = temp['price']
    Options_m0['Spotprice'] = Spot_price
   ###################################################################
   # 定位ATM K
  
    call_option = Options_m0[Options_m0['Type'] == "c"]
    put_option = Options_m0[Options_m0['Type'] == "p"]
    
    diff = np.abs(call_option['Strike'].values - call_option['Spotprice'].values)
    locc = np.where(diff == np.min(diff))[0][-1]  # 若相邻两个K一致，取大者
    atm_call = call_option.index[locc]
    atm_put = put_option.index[locc]
    
    call_ids = call_option.index
    put_ids = put_option.index
    k_list = call_option.loc[call_ids, 'Strike'].to_list()
    atm_k = call_option.loc[atm_call, 'Strike']
    
    call_prices = call_option.loc[call_ids, 'Close_price'].to_list()
    put_prices = put_option.loc[put_ids, 'Close_price'].to_list()
    
    #index.append(Futures_code[0])
    # else:
    #     temp = df_V.drop(Futures_code,1)
    #     temp = temp.drop('max_idx',1)
    #     temp['secondmax_idx'] = temp.idxmax(axis=1)
    #     Futures_code_new = temp['secondmax_idx'][i]
    #     Futures_price_new = df_P[Futures_code_new][i]
    #     options = df_list[df_list['Futures'] == Futures_code_new[0]]
    #     contracts = list(options["Code"])
    #     options.index = options['Code']
    #     options = options[['Type','Strike','EXE_ENDDATE']]
    #     data_options = w.wsd(contracts,"close",day,day,"")
    #     temp = pd.DataFrame(data_options.Data).T
    #     temp.index = data_options.Codes
    #     temp.columns = ['price']
    #     options['Close_price'] = temp['price']
    #     options['F'] = Futures_price_new
    #     index.append(Futures_code_new[0])
    
    
    t = (w.tdayscount(date, Options_m0['EndDate'].iloc[0], "").Data[0][0]-1)/245
    
    syn_f = calc_f(call_prices, put_prices, call_ids, atm_call, k_list, r, t)
    # F = syn_f
######利用拟合计算skew ############       
    ####计算IV,GREEKS
    df = Cal_greeks(date,Options_m0)

    ####拟合IV曲线
    curve = calc_vol_surface(df)
    
    ###计算skew
    #skew.append((curve[13]-curve[3])/curve[8])
    skew.append((curve[3]-curve[8])) 
    # delta 小于 0.5的 是 call iv， 大于0.5的是 put iv。0-8为otm_call，9-12为otm_put
    
###### 上述依旧在for循环里面######

    
skew = pd.DataFrame(skew)
skew.index = df_P['Date']
skew.columns=['skew']
#skew['Futures_code'] = pd.DataFrame(index,index=df_V.index)
skew.to_excel('Cskew.xls')

