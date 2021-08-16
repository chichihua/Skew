# -*- coding: utf-8 -*-


import pandas as pd
from PyQt5.QtWidgets import *
from WindPy import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import numpy_indexed as npi
import py_vollib.black.implied_volatility as biv
import py_vollib.black.greeks.analytical as bga
import pandas_market_calendars as mcal
#import Lib_OptionCalculator as OC

w.start()


def form(date):
    from dateutil.parser import parse
    return parse(str(date)).strftime('%Y/%m/%d')


def LoadData2(symbols):
    error, data2 = w.wss(symbols, "exe_price,exe_enddate,exe_mode,exe_ratio", usedf=True)
    data2 = data2.replace('认购', 'call')
    data2 = data2.replace('认沽', 'put')
    data2['EXE_ENDDATE'] = data2['EXE_ENDDATE'].apply(form)
    return data2


def LoadData3(underlying):
    field = "exchange=sse;windcode="+underlying+";status=trading;field=wind_code,exercise_price,expire_date,call_or_put,contract_unit"
    wset_data = w.wset("optioncontractbasicinfo", field)
                       #"exchange=sse;windcode=510300.SH;status=trading;field=wind_code,exercise_price,expire_date,call_or_put,contract_unit")
    df = pd.DataFrame(wset_data.Data, index=wset_data.Fields)
    df1 = df.T
    df1 = df1.replace('认沽', 'put')
    df1 = df1.replace('认购', 'call')
    df1['wind_code'] = df1['wind_code'] + '.SH'
    contracts = list(df1['wind_code'])
    df1.set_index(['wind_code'], inplace=True)
    df1.rename(columns={'exercise_price': 'EXE_PRICE', 'expire_date': 'EXE_ENDDATE', 'call_or_put': 'EXE_MODE',
                        'contract_unit': 'EXE_RATIO'}, inplace=True)
    df1['EXE_ENDDATE'] = df1['EXE_ENDDATE'].apply(form)
    return df1, contracts


###################2020.09.24改，用每个到期日所有strike的syn price的median
def CalSynFuturePrice(df): ### 计算 合成现货价格
    df = df[['EXE_ENDDATE', 'EXE_PRICE', 'SETTLE', 'EXE_MODE']]
    flag = {'call': 1, 'put': -1}
    df['EXE_MODE'] = df['EXE_MODE'].map(flag)
    if len(df) % 2 == 0:
        df = np.array(df)
        # 标记期权type
       # temp = npi.group_by(df[:, -1]).split(df[:, :])
       # temp1 = np.insert(temp[0], -1, 1, axis=1)
       # temp2 = np.insert(temp[1], -1, -1, axis=1)
       # temp3 = np.vstack([temp1, temp2])
       # pre_syn = temp3[:, 2] * temp3[:, 3] + 0.5 * temp3[:, 1]
        pre_syn = df[:, 2] * df[:, 3] + 0.5 * df[:, 1]
       # df1 = np.insert(temp3[:, 0:2], 2, values=pre_syn, axis=1).astype(float)
        df1 = np.insert(df[:, 0:2], 2, values=pre_syn, axis=1).astype(float)
        # 计算每个strike下的F
        temp4 = npi.group_by(df1[:, 1]).sum(df1[:, 2])
        F = np.median(temp4[1])
    else:
        F = np.nan
        print('### data missing for syn_fwd calculation ###')
    return F
#########################
# 五个最小的, no more use
def CalSynFuturePrice2(TC1, TP1):
    premium1 = list(TC1['S_DQ_CLOSE'])
    premium2 = list(TP1['S_DQ_CLOSE'])
    diff = abs(pd.Series(premium1) - pd.Series(premium2))
    index = diff.nsmallest(5).index.values

    p = []
    for i in index:
        p.append(premium1[i] + TC1.values[i, 4] - premium2[i])
    PriceSyn = sum(p) / 5
    return PriceSyn


def RemainingDays(enddate):
    num = 245
    today = datetime.today()
    days = w.tdayscount(today, enddate, "")
    days = days.Data[0][0]

    h = datetime.now().hour
    m = datetime.now().minute
    s = datetime.now().second
    t = h + m / 60 + s / 3600
    if t < 9.5:
        t0 = 4
    if 9.5 <= t < 11.5:
        t0 = 2 + 11.5 - t
    if 11.5 <= t < 13:
        t0 = 2
    if 13 <= t < 15:
        t0 = 15 - t
    if t >= 15:
        t0 = 0

    return (int(days) - 1 + t0 / 4) / num

# no more use
def RemainingDays2(startdate, enddate, dfin):
    num = 245
    start = dfin[dfin['date'] == startdate].index.values
    end = dfin[dfin['date'] == enddate].index.values
    return float(end - start) / num


def CalSynVol_Call(dataCallInput_df, tagetDelta, strikeNa='EXE_PRICE', IV_Na='Implied Volatility', deltaNa='DELTA'):
    """
    计算call option的合成IV
    """

    # 按照行权价从低往高往下排列
    dataCall_df = dataCallInput_df.sort_values(by=strikeNa)
    dataCall_df.reset_index(drop=True, inplace=True)
    diff = dataCall_df[deltaNa] - tagetDelta

    # get positive parts
    a = (diff > 0) * diff
    if 0 < sum(diff > 0) < len(diff):
        index1 = sum(a > 0)
        index2 = index1 - 1
    else:
        if sum(diff > 0) == len(diff):
            index1 = len(diff) - 1
            index2 = index1 - 1
        else:
            index1 = 1
            index2 = 0

    x1 = dataCall_df[deltaNa].iloc[index1]
    x2 = dataCall_df[deltaNa].iloc[index2]
    y1 = dataCall_df[IV_Na].iloc[index1]
    y2 = dataCall_df[IV_Na].iloc[index2]

    if x1 == x2:
        return y1
    IVSyn = y1 + (tagetDelta - x1) * (y1 - y2) / (x1 - x2)

    return IVSyn


def CalSynVol_Put(dataPutInput_df, tagetDelta, strikeNa='EXE_PRICE', IV_Na='Implied Volatility', deltaNa='DELTA'):
    """
    计算put option的合成IV
    """
    # 按照行权价从低往高往下排列
    dataPut_df = dataPutInput_df.sort_values(by=strikeNa)
    dataPut_df.reset_index(drop=True, inplace=True)
    diff = tagetDelta - dataPut_df[deltaNa]

    a = (diff > 0) * diff
    if 0 < sum(diff > 0) < len(diff):
        index1 = sum(a == 0)
        index2 = index1 - 1
    else:
        if sum(diff > 0) == len(diff):
            index1 = 1
            index2 = 0
        else:
            index1 = len(diff) - 1
            index2 = index1 - 1

    y1 = dataPut_df[IV_Na].iloc[index1]
    y2 = dataPut_df[IV_Na].iloc[index2]
    x1 = dataPut_df[deltaNa].iloc[index1]
    x2 = dataPut_df[deltaNa].iloc[index2]

    if x1 == x2:
        return y1

    IVSyn = y1 + (tagetDelta - x1) * (y1 - y2) / (x1 - x2)
    return IVSyn


def CalStraddle(TC, TP, deltaNa='DELTA', optPrice='OPT_PRICE'):
    """
    :param TC: DataFrame,CalGreeks函数的返回值
    :param TP: DataFrame,CalGreeks函数的返回值
    :param deltaNa: str,代表列名
    :param optPrice: str,代表列名
    :return: float
    """
    # 找到delta最接近0.5的两个索引,标记为index1和index2
    twoIndex = abs(TC[deltaNa] - 0.5).nsmallest(2).index.values
    index1, index2 = twoIndex[0], twoIndex[1]

    # 线性插值
    x0 = TC[deltaNa].loc[index1]
    x1 = TC[deltaNa].loc[index2]
    y0 = TC[optPrice].loc[index1]
    y1 = TC[optPrice].loc[index2]
    CallATM = (0.5 - x1) * (y0 - y1) / (x0 - x1) + y1

    # put计算同上
    twoIndex = abs(TP[deltaNa] + 0.5).nsmallest(2).index.values
    index1, index2 = twoIndex[0], twoIndex[1]

    x0 = TP[deltaNa].loc[index1]
    x1 = TP[deltaNa].loc[index2]
    y0 = TP[optPrice].loc[index1]
    y1 = TP[optPrice].loc[index2]
    PutATM = (-0.5 - x1) * (y0 - y1) / (x0 - x1) + y1
    Straddle = CallATM + PutATM

    return Straddle




###############2020.09.24加 计算隐含波动率
def my_implied_vol(S, F, K, r, T, option_type):
    try:
        my_impVol = biv.implied_volatility_of_discounted_option_price(S, F, K, r,T, option_type.lower())
    except:
        my_impVol = np.nan
    return my_impVol
##########################

###################2020.09.24加 vollib计算greeks
def calc_basic_greeks(df,F,T):
    if df.empty:
        df = df.reindex(columns=df.columns.tolist() + ['delta', 'gamma', 'vega', 'theta'])
        return df
    df['DELTA'] = df.apply(lambda row: bga.delta(row['EXE_MODE'].lower(), F, row['EXE_PRICE'], T, 0, row['Implied Volatility']), axis=1)
    df['GAMMA'] = df.apply(lambda row: bga.gamma(row['EXE_MODE'].lower(), F, row['EXE_PRICE'], T, 0, row['Implied Volatility']), axis=1)
    df['VEGA'] = df.apply(lambda row: bga.vega(row['EXE_MODE'].lower(), F, row['EXE_PRICE'], T, 0, row['Implied Volatility']), axis=1)
    df['THETA'] = df.apply(lambda row: bga.theta(row['EXE_MODE'].lower(), F, row['EXE_PRICE'], T, 0, row['Implied Volatility'])*365/245, axis=1)
    return df
##########################

##################2020.09.24改，整合greeks
def CalGreeks(datac, datap):

    data1c = datac#.copy()
    data1p = datap#.copy()
    datac['SETTLE'] = (data1c['RT_ASK1'] + data1c['RT_BID1']) / 2
    datap['SETTLE'] = (data1p['RT_ASK1'] + data1p['RT_BID1']) / 2
    #datac['SETTLE'].loc[(datac['RT_BID1'] - datac['RT_ASK1'] > 0.01) & (datac["EXE_PRICE"] < datac['INDEX_PRICE'])] = np.nan
    #datap['SETTLE'].loc[(datap['RT_BID1'] - datap['RT_ASK1'] > 0.01) & (datap["EXE_PRICE"] > datap['INDEX_PRICE'])] = np.nan


    df = pd.concat([datac, datap], axis=0)
    strike = df['EXE_PRICE'].loc[df['SETTLE'].isna()]
    for i in strike:
        df = df[df["EXE_PRICE"] != i]
    # df['EXE_ENDDATE'] = pd.to_datetime(df['EXE_ENDDATE'])
    df['EXE_ENDDATE'] = df['EXE_ENDDATE'].apply(lambda x: datetime.strftime(datetime.strptime(str(x), "%Y/%m/%d"), "%Y%m%d"))


    F = CalSynFuturePrice(df)
    T = RemainingDays(df['EXE_ENDDATE'].iloc[0])
    r = 0
    flag = {'call': 'c', 'put': 'p'}
    df['EXE_MODE'] = df['EXE_MODE'].map(flag)
    #计算隐含波动率
    #can modify here to include bidIV and askIV
    df['Implied Volatility'] = df.apply(lambda row: my_implied_vol(row['SETTLE'], F, row['EXE_PRICE'], r, T, row['EXE_MODE']), axis=1)

    df.loc[df['Implied Volatility'] < 0, 'Implied Volatility'] = np.nan
    pd.options.mode.chained_assignment = None  # default='warn'
    strike = df['EXE_PRICE'].loc[df['Implied Volatility'].isna()]
    for i in strike:
       df = df[df["EXE_PRICE"] != i]
    #P = w.wsq("510300.SH", "rt_last").Data
    # strike > F 的合约
    otmc = df[(df['EXE_MODE'] == 'c') & (df.EXE_PRICE > F)]
    #otmc = df[(df['EXE_MODE'] == 'c') & (df.EXE_PRICE > P)]
    otmc = calc_basic_greeks(otmc,F,T)
    itmp = df[(df['EXE_MODE'] == 'p') & (df.EXE_PRICE > F)]
    #itmp = df[(df['EXE_MODE'] == 'p') & (df.EXE_PRICE > P)]

    if len(otmc) != len(itmp):
        print('###please check data from greeks calculation,  otmp does not match in length of itmp###')
    rhs_options = otmc.append(itmp, sort=False)[otmc.columns.tolist()].sort_values(['EXE_PRICE'])

    if not rhs_options.empty:
        rhs_options['DELTA'] = rhs_options['DELTA'].fillna(method='ffill')
        rhs_options['GAMMA'] = rhs_options['GAMMA'].fillna(method='ffill')
        rhs_options['VEGA'] = rhs_options['VEGA'].fillna(method='ffill')
        rhs_options['THETA'] = rhs_options['THETA'].fillna(method='ffill')
        itmp = rhs_options[(rhs_options['EXE_MODE'] == 'p') & (rhs_options.EXE_PRICE > F)]
        #itmp = rhs_options[(rhs_options['EXE_MODE'] == 'p') & (rhs_options.EXE_PRICE > P)]
        itmp['DELTA'] = itmp['DELTA'] - 1
        rhs_options = otmc.append(itmp, sort=False)[otmc.columns.tolist()].sort_values(['EXE_PRICE'])

    #strike < F 的合约
    otmp = df[(df['EXE_MODE'] == 'p') & (df.EXE_PRICE <= F)]
    #otmp = df[(df['EXE_MODE'] == 'p') & (df.EXE_PRICE <= P)]
    otmp = calc_basic_greeks(otmp,F,T)
    itmc = df[(df['EXE_MODE'] == 'c') & (df.EXE_PRICE <= F)]
    #itmc = df[(df['EXE_MODE'] == 'c') & (df.EXE_PRICE <= P)]

    if len(otmp) != len(itmc):
        print('###please check data from greeks calculation,  otmp does not match in length of itmp###')
    lhs_options = otmp.append(itmc, sort=False)[otmp.columns.tolist()].sort_values(['EXE_PRICE'])
    if not lhs_options.empty:
        lhs_options['DELTA'] = lhs_options['DELTA'].fillna(method='ffill')
        lhs_options['GAMMA'] = lhs_options['GAMMA'].fillna(method='ffill')
        lhs_options['VEGA'] = lhs_options['VEGA'].fillna(method='ffill')
        lhs_options['THETA'] = lhs_options['THETA'].fillna(method='ffill')
        itmc = lhs_options[(lhs_options['EXE_MODE'] == 'c') & (lhs_options.EXE_PRICE <= F)]
        #itmc = lhs_options[(lhs_options['EXE_MODE'] == 'c') & (lhs_options.EXE_PRICE <= P)]
        itmc['DELTA'] = itmc['DELTA'] + 1
        lhs_options = otmp.append(itmc, sort=False)[otmp.columns.tolist()].sort_values(['EXE_PRICE'])

    df = lhs_options.append(rhs_options, sort=False)[lhs_options.columns.tolist()].sort_values(['EXE_PRICE'])
    df.insert(0, 'Contract', list(df.index))
    df.index = [i for i in range(len(df))]
    df['F'] = F
    df['T'] = T
    return df
########################



