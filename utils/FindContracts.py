import numpy as np
import pandas as pd
from WindPy import w
from chinese_calendar import is_workday, is_holiday
from datetime import datetime, timedelta
from dateutil.relativedelta import *


def get_opt_info(underlying='510050.SH'):
    cols = ['代码', '简称', '标的代码', '行权价', '行权日', '交割月份']
    data = w.wset("optioncontractbasicinformation",
                  "exchange=sse;"
                  "windcode=%s;" % underlying +
                  "status=trading;"
                  "field=wind_code,sec_name,option_mark_code,exercise_price,exercise_date,limit_month")
    data = pd.DataFrame(data=np.array(data.Data).T, columns=cols)
    return data


class WeekdayLocator:

    def __init__(self, year: int, month: int, i_th=4, weekday=3):
        self.year = year
        self.month = month
        self.i_th = i_th
        self.weekday = weekday
        self.start_date = None
        self.end_date = None

    def generate_dates_from_range(self):
        # 根据month生成月起始日到月终止日的所有日期
        self.start_date = datetime.strptime('%s-%s-1' % (self.year, self.month), '%Y-%m-%d')
        self.end_date = datetime.strptime(
            '%s-%s-1' % (self.year + (self.month/12 == 1), (self.month/12 != 1) * self.month + 1), '%Y-%m-%d'
        ) + timedelta(-1)
        date_list = [d.to_pydatetime() for d in list(pd.date_range(start=self.start_date, end=self.end_date))]
        return date_list

    def locate_weekday(self):
        date_list = self.generate_dates_from_range()
        # 构建日期DataFrame，获取日期对应星期
        date_df = pd.DataFrame()
        date_df['date'] = date_list
        date_df['weekday'] = date_df['date'].apply(lambda x: x.weekday()+1)
        date_df = date_df.set_index('date', drop=True)
        date_df.index = pd.to_datetime(date_df.index)

        located_date = date_df[date_df['weekday'] == self.weekday].index[self.i_th - 1].date()
        return located_date

    def adjust_with_chinese_holidays(self):
        located_date = self.locate_weekday()
        # w.start()
        trading_dates = w.tdays(self.start_date, self.end_date, "").Data[0]

        if is_holiday(located_date) and located_date not in trading_dates:
            increment = 1
            while located_date + timedelta(days=increment) not in trading_dates:
                increment += 1
            assert is_workday(located_date + timedelta(days=increment))
            located_date += timedelta(days=increment)
            print('Skip', located_date + timedelta(days=-increment), 'to', located_date)

        return datetime.strptime(str(located_date), '%Y-%m-%d')

    def locate(self):
        return self.adjust_with_chinese_holidays()


class FindContracts(object):

    def __init__(self, date, close_info, contract_info, underlying='510050.SH', n1=6, n2=5, n3=4):
        # w.start()
        if type(date) != datetime:
            self.date = datetime.strptime(date, '%Y-%m-%d')
        else:
            self.date = date
        self.close_info = close_info
        self.contract_info = contract_info
        self.contract_info['行权日'] = pd.to_datetime(self.contract_info['行权日'])
        self.underlying = underlying
        # self.s = close_info.loc[date, underlying]
        self.s = w.wsd(self.underlying, 'close', self.date, self.date, '').Data[0][0]
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.maturities = None

    def locate_4m(self):
        # 定位当月、次月、季月、次季月对应合约到期日
        current_month_maturity = WeekdayLocator(self.date.year, self.date.month).locate()
        if self.date < current_month_maturity:  # 未到当月行权日，以当月到期合约为当月合约
            m00 = current_month_maturity
        else:  # 已过当月行权日，以下月到期合约为当月合约
            m00 = WeekdayLocator(
                self.date.year + (self.date.month/12 == 1),
                (self.date.month/12 != 1) * self.date.month + 1
            ).locate()
        m01 = WeekdayLocator(
            m00.year + (m00.month/12 == 1),
            (m00.month/12 != 1) * m00.month + 1
        ).locate()
        m02 = WeekdayLocator(
            m01.year + (m01.month/3 == 4),
            ((m01.month/3 != 4) * int(m01.month/3) + 1) * 3
        ).locate()
        # 季月后推3个月得到次季月，此处直接调用dateutil.relativedelta
        m03 = WeekdayLocator(
            (m02 + relativedelta(months=3)).year,
            (m02 + relativedelta(months=3)).month
        ).locate()
        return [m00, m01, m02, m03]

    def select_ids_by_n(self, input_ids: pd.Index, loc, n):
        # 按照档位N选取合约，若合约数量不足则取到能取得档位
        atm_id = input_ids[loc]
        output_ids = []
        output_ids += input_ids[max(loc-n, 0):loc].to_list()
        output_ids += input_ids[loc:loc+n+1].to_list()
        return output_ids, atm_id

    def find_contract_ids(self, maturity):
        # S在3以下取上下N1= 6档
        # 5以下取上行N2 = 5档
        # 5以上取上下N3 = 4档
        if self.s <= 3:
            n = self.n1
        elif 3 < self.s <= 5:
            n = self.n2
        else:
            n = self.n3
        # 定位行权日该标的所有行权价的call、put合约，并根据行权价升序排列
        calls = self.contract_info[
            (self.contract_info['行权日'] == pd.Timestamp(maturity.date()))
            & (self.contract_info['标的代码'] == int(self.underlying.replace('.SH', '')))
            & (self.contract_info['简称'].str.contains('购'))
            ].sort_values(by='行权价', ascending=True)
        calls = calls[~calls.index.duplicated(keep='first')]  # 确保无重复期权ID使wsd函数报错
        puts = self.contract_info[
            (self.contract_info['行权日'] == pd.Timestamp(maturity.date()))
            & (self.contract_info['标的代码'] == int(self.underlying.replace('.SH', '')))
            & (self.contract_info['简称'].str.contains('沽'))
            ].sort_values(by='行权价', ascending=True)
        puts = puts[~puts.index.duplicated(keep='first')]  # 确保无重复期权ID使wsd函数报错
        assert all(calls['行权价'].values == puts['行权价'].values)  # assert call、put合约行权价一致
        # 定位ATM K
        diff = np.abs(calls['行权价'].values - self.s)
        loc = np.where(diff == np.min(diff))[0][-1]  # 若相邻两个K一致，取大者

        call_ids, atm_call = self.select_ids_by_n(calls.index, loc, n)
        put_ids, atm_put = self.select_ids_by_n(puts.index, loc, n)
        k_list = calls.loc[call_ids, '行权价'].to_list()
        atm_k = calls.loc[atm_call, '行权价']
        assert len(call_ids) == len(put_ids) == len(k_list)
        return call_ids, put_ids, atm_call, atm_put, k_list, atm_k

    # def filter_maturities(self, tt):
    #     # 剩余交易天数 < tt: 剔除该月份合约
    #     for m in self.maturities:
    #         if w.tdayscount(self.date, m, "").Data[0][0] < tt:
    #             self.maturities.remove(m)
    #             del self.names[m]
    #     assert len(self.names) == len(self.maturities)
