from utils.FindContracts import *
import numpy as np
import pandas as pd
from WindPy import w
import py_vollib.black.implied_volatility as biv
import py_vollib.black_scholes.implied_volatility as bsiv
import py_vollib.black.greeks.analytical as bga
import py_vollib.black_scholes.greeks.analytical as bsga
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from scipy.optimize import curve_fit
# import matplotlib.ticker as ticker
import copy


class IVCurve(FindContracts):

    def __init__(self, date, maturities, close_info, contract_info, underlying='510050.SH', exclude_a=True, tt=0,
                 n1=6, n2=5, n3=4, synfutures_n=2, method='bs', r_bs=0.03, r_black=0):
        super(IVCurve, self).__init__(date, close_info, contract_info, underlying, exclude_a, tt, n1, n2, n3)
        self.synfutures_n = synfutures_n  # 计算合成期货时使用上下几档
        self.method = method  # 'bs' or 'black'
        if self.method == 'bs':
            self.r = r_bs
        elif self.method == 'black':
            self.r = r_black
        self.maturities = maturities
        # self.maturities = self.locate_4m()  # 定位当月、次月、季月、次季月对应合约到期日
        # self.maturities = [m.replace(hour=15, minute=0, second=0) for m in self.maturities]  # 到期日时分秒切换到收盘时间
        self.names = {}
        for i in range(len(self.maturities)):
            self.names[self.maturities[i]] = '%iM' % i
        self.atm_ks = {}
        self.call_prices = {}
        self.put_prices = {}
        self.ts = {}
        self.fs = {}
        self.strike_curves = {}
        # self.fitted_strike_curves = {}
        self.greeks = {}
        self.delta_curves = {}
        # self.fitted_delta_curves = {}
        # self.call_wings = {}
        # self.put_wings = {}
        # self.fitted_call_wings = {}
        # self.fitted_put_wings = {}
        self.atm_ivs = {}
        self.delta_call_ivs = {}
        self.delta_put_ivs = {}
        self.skew = {}

    def get_opt_prices(self, call_ids, put_ids):
        call_prices = np.array(w.wsd(call_ids, "close", self.date, self.date, "").Data[0])
        put_prices = np.array(w.wsd(put_ids, "close", self.date, self.date, "").Data[0])
        return call_prices, put_prices

    def calc_t(self, maturity, annual_trading_days=245):
        # 计算合约距到期日剩余交易天数/年交易天数
        # return w.tdayscount(self.date, maturity, "").Data[0][0] / annual_trading_days
        days = w.tdayscount(self.date, maturity, "").Data[0][0]
        h = self.date.hour
        m = self.date.minute
        s = self.date.second
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
        return (int(days) - 1 + t0 / 4) / annual_trading_days

    def calc_f(self, call_prices, put_prices, call_ids, atm_call, k_list, r, t):
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
        cap = self.synfutures_n * 2 + 1
        loc = np.where(np.array(call_ids) == atm_call)[0][0]
        # print(call_prices[loc], put_prices[loc], k_list[loc])
        try:
            for n in range(len(call_prices)):
                if n == 0:
                    if np.isnan(call_prices[loc]) == False \
                            and np.isnan(put_prices[loc]) == False \
                            and np.isnan(k_list[loc]) == False:
                        # S = C + K*e^{-r*t} - P
                        total += call_prices[loc] - put_prices[loc] + k_list[loc] * np.exp(- r * t)
                        count += 1
                        if count == cap:
                            break
                    else:
                        cap = self.synfutures_n * 2  # 若ATM价格为nan，取两边
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
                                count += 2
                                if count == cap:
                                    break
                    except:
                        pass

            # assert count >= 3
            f = total / count
        except:
            f = np.nan
        return f

    def call(self, func, args):
        # 打包参数，以便调用不同method下对应函数
        try:
            output = func(*args)
        except Exception as ex:
            print(ex)
            output = np.nan
        return output

    def calc_iv(self, s, f, k, t, flag):
        r = self.r
        if self.method == 'black':
            func = biv.implied_volatility_of_discounted_option_price
            args = [s, f, k, r, t, flag]
        elif self.method == 'bs':
            func = bsiv.implied_volatility
            args = [s, f, k, t, r, flag]

        iv = self.call(func, args)
        return iv

    def calc_greeks(self, flag, f, k, t, iv):
        r = self.r
        if self.method == 'black':
            func = bga
        elif self.method == 'bs':
            func = bsga
        args = [flag, f, k, t, r, iv]

        delta = self.call(func.delta, args)
        gamma = self.call(func.gamma, args)
        vega = self.call(func.vega, args)
        theta = self.call(func.theta, args)
        return delta, gamma, vega, theta

    def calc_strike_curve_and_greeks(self, maturity):
        # 计算同一到期日不同行权价的虚值合约的iv, greeks
        call_ids, put_ids, atm_call, atm_put, k_list, atm_k = self.find_contract_ids(maturity)
        self.atm_ks[maturity] = atm_k
        call_prices, put_prices = self.get_opt_prices(call_ids, put_ids)
        self.call_prices[maturity] = call_prices
        self.put_prices[maturity] = put_prices
        assert len(call_prices) == len(put_prices) == len(call_ids)
        t = self.calc_t(maturity)
        self.ts[maturity] = t
        f = self.calc_f(call_prices, put_prices, call_ids, atm_call, k_list, self.r, t)
        self.fs[maturity] = f
        # print('ATM K = %.4f' % atm_k)
        # print('t = %.4f' % t)
        # print('F = %.4f' % f)
        strike_curve = pd.Series(index=k_list, name='IV')
        greeks = pd.DataFrame(columns=['K', 'Delta', 'Gamma', 'Vega', 'Theta'])
        # 使用虚值期权价格计算合约iv, greeks
        # 等于atm k，atm iv <- (atm call iv + atm put iv) / 2, 保留call, put两个合约
        # 大于atm k, iv <- call iv
        # 小于atm k, iv <- put iv
        index = 0
        for i in range(len(k_list)):
            append_row = pd.DataFrame(index=[i], columns=['K', 'Delta', 'Gamma', 'Vega', 'Theta'])
            append_row['K'] = k_list[i]
            if k_list[i] == atm_k:
                call_iv = self.calc_iv(call_prices[i], f, k_list[i], t, 'c')
                put_iv = self.calc_iv(put_prices[i], f, k_list[i], t, 'p')
                strike_curve.iloc[i] = (call_iv + put_iv) / 2
                # strike_curve[k_list[i]] 改为 strike_curve.iloc[i]: 按整数index写入series，防止K一致时索引至两行值
                # atm put greeks
                append_row['Delta'], append_row['Gamma'], append_row['Vega'], append_row['Theta'] = self.calc_greeks(
                    'p', f, k_list[i], t, strike_curve.iloc[i])
                greeks = greeks.append(append_row, ignore_index=True)
                # atm call greeks
                append_row['Delta'], append_row['Gamma'], append_row['Vega'], append_row['Theta'] = self.calc_greeks(
                    'c', f, k_list[i], t, strike_curve.iloc[i])
                greeks = greeks.append(append_row, ignore_index=True)
            elif k_list[i] > atm_k:
                strike_curve.iloc[i] = self.calc_iv(call_prices[i], f, k_list[i], t, 'c')
                # print('K = %.4f , Call Price = %.4f, ID: %s' % (k_list[i], call_prices[i], call_ids[i]))
                append_row['Delta'], append_row['Gamma'], append_row['Vega'], append_row['Theta'] = self.calc_greeks(
                    'c', f, k_list[i], t, strike_curve.iloc[i])
                greeks = greeks.append(append_row, ignore_index=True)
                # print('')  # debug line
            elif k_list[i] < atm_k:
                strike_curve.iloc[i] = self.calc_iv(put_prices[i], f, k_list[i], t, 'p')
                # print('K = %.4f , Put Price = %.4f , ID: %s' % (k_list[i], put_prices[i], put_ids[i]))
                append_row['Delta'], append_row['Gamma'], append_row['Vega'], append_row['Theta'] = self.calc_greeks(
                    'p', f, k_list[i], t, strike_curve.iloc[i])
                greeks = greeks.append(append_row, ignore_index=True)
                # print('')  # debug line
        strike_curve = strike_curve.dropna()
        # strike_curve = strike_curve.sort_index(ascending=True)
        greeks = greeks.dropna()
        return strike_curve, greeks

    def get_strike_curves_and_greeks(self):
        # 根据当前日期对应的当月、次月、季月、次季月合约，调用calc_strike_curve_and_greeks生成对应的strike curve和greeks
        for m in self.maturities:
            try:
                self.strike_curves[m], self.greeks[m] = self.calc_strike_curve_and_greeks(m)
            except (ValueError, AttributeError):
                self.strike_curves[m], self.greeks[m] = None, None
                # print('')  # debug line

    def calc_delta_curve(self, maturity):
        f = self.fs[maturity]
        t = self.ts[maturity]
        # k = self.greeks[maturity].drop_duplicates(subset='K', keep='first')['K'].to_numpy()
        # y = pd.DataFrame(self.strike_curves[maturity].values ** 2 * t, columns=['y'])
        # y = y.drop_duplicates(subset='y', keep='first')['y'].to_numpy()

        # y (IV) 需确保与log_m维度一致，否则OLS矩阵运算报错
        k = copy.deepcopy(self.greeks[maturity])
        y = copy.deepcopy(self.strike_curves[maturity])
        k['iv'] = np.nan
        for index in y.index:
            if type(y.loc[index]) == np.float64:
                k.loc[k['K'] == index, 'iv'] = y.loc[index]
            else:
                for i in range(len(y.loc[index])):
                    k.loc[k[k['K'] == index].index[i], 'iv'] = y.loc[index].iloc[i]

        k['y'] = k['iv'].apply(lambda x: x ** 2 * t)
        y = k['y'].values
        log_m = np.log(k['K'].values / f)
        assert len(log_m) == len(y)

        X = np.stack([np.ones(len(log_m)), log_m, log_m**2], axis=1)

        df = pd.DataFrame(columns=['K', 'log_moneyness', 'Delta', 'IV', 'parameterized_IV'])
        df['K'] = k['K'].values
        df['log_moneyness'] = log_m
        df['Delta'] = np.abs(k['Delta'].values)
        df['IV'] = k['iv'].values

        try:
            model = OLS(y, X)
            results = model.fit()
            parameterized_iv = np.sqrt(np.dot(X, results.params) / t)
            df['parameterized_IV'] = parameterized_iv
        except:
            df['parameterized_IV'] = np.nan

        func = (lambda x, a, b, c: a * x ** 2 + b * x + c)
        deltas = np.round(np.arange(0.1, 0.9+0.05, 0.05), 2)

        try:
            df.drop(df[(np.abs(df['Delta']) < 0.1) | (np.abs(df['Delta']) > 0.9)].index, inplace=True)
            df.dropna(inplace=True)
            popt, pcov = curve_fit(func, df['Delta'].to_numpy(), df['parameterized_IV'].to_numpy(),
                                   bounds=([0, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))
            curve = func(deltas, *popt)
            curve = pd.Series(index=deltas, data=curve)
        except:
            curve = pd.Series(index=deltas, data=np.nan)
        return curve

    def get_delta_curves(self, plot=False):
        if plot:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlim(0, 1)
        for m in self.maturities:
            try:
                self.delta_curves[m] = self.calc_delta_curve(m)
                if plot:
                    ax.plot(self.delta_curves[m].index, self.delta_curves[m].values, label=self.names[m])
            except KeyError:
                self.delta_curves[m] = None
        if plot:
            ax.set_title('%s Delta Curve' % datetime.strftime(self.date, '%Y-%m-%d %H:%M:%S'))
            ax.set_xlabel('Delta')
            ax.set_ylabel('IV')
            ax.legend(loc='lower right')
            plt.show()

    # def cubic_spline_interpolate(self, curve: pd.Series):
    #     # 三次样条插值拟合曲线
    #     curve = curve.sort_index(ascending=True)
    #     return CubicSpline(curve.index.to_numpy(), curve.to_numpy())

    # def get_fitted_strike_curves(self, plot=False):
    #     if plot:
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #     for m in self.maturities:
    #         x = self.strike_curves[m].index.to_numpy()
    #         y = self.strike_curves[m].to_numpy()
    #         cs = self.cubic_spline_interpolate(self.strike_curves[m])
    #         self.fitted_strike_curves[m] = cs
    #         # print('')
    #         if plot:
    #             xs = np.arange(np.min(x), np.max(x), 0.01)
    #             ax.plot(x, y, label=self.names[m])
    #             ax.plot(xs, cs(xs), linestyle='dotted', label='%s: cs' % self.names[m])
    #     if plot:
    #         ax.set_title('%s Strike Curve' % datetime.strftime(self.date, '%Y-%m-%d'))
    #         ax.set_xlabel('K')
    #         ax.set_ylabel('IV')
    #         ax.legend(loc='lower right')
    #         plt.show()

    # def calc_delta_curve(self, maturity, interval=0.1):
    #     # 基于三次样条拟合的strike curve，根据对应interval的strikes，计算iv及delta，得到离散delta curve后再进行三次样条拟合
    #     strikes = np.round(np.arange(np.min(self.strike_curves[maturity].keys()),
    #                                  np.max(self.strike_curves[maturity].keys()) + interval, interval), 2)
    #     ivs = self.fitted_strike_curves[maturity](strikes)
    #     f = self.fs[maturity]
    #     t = self.ts[maturity]
    #     deltas = []
    #     for i in range(len(strikes)):
    #         if strikes[i] >= self.atm_ks[maturity]:
    #             deltas.append(self.calc_greeks('c', f, strikes[i], t, ivs[i])[0])
    #         else:
    #             deltas.append(self.calc_greeks('p', f, strikes[i], t, ivs[i])[0])
    #     delta_curve = pd.Series(index=deltas, data=ivs)
    #     delta_curve = delta_curve.sort_index()
    #     return delta_curve[delta_curve.index >= 0], delta_curve[delta_curve.index < 0]
    #
    # def get_delta_curves(self, interval=0.1):
    #     for m in self.maturities:
    #         self.call_wings[m], self.put_wings[m] = self.calc_delta_curve(m, interval)

    # def get_fitted_delta_curves(self, plot=False):
    #     if plot:
    #         i = 0
    #         colors = ['red', 'green', 'blue', 'yellow']
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         ax2 = ax.twiny()
    #         ax.set_xlim(0, 1)
    #         ax2.set_xlim(-1, 0)
    #     for m in self.maturities:
    #         csc = self.cubic_spline_interpolate(self.call_wings[m])
    #         self.fitted_call_wings[m] = csc
    #         csp = self.cubic_spline_interpolate(self.put_wings[m])
    #         self.fitted_put_wings[m] = csp
    #         if plot:
    #             xc = self.call_wings[m].index.to_numpy()
    #             yc = self.call_wings[m].to_numpy()
    #             xp = self.put_wings[m].index.to_numpy()
    #             yp = self.put_wings[m].to_numpy()
    #             # xcs = np.arange(0.1, 0.5+0.001, 0.001)
    #             # xps = np.arange(-0.5, -0.1+0.001, 0.001)
    #             # ychat = csc(xcs)
    #             # yphat = csp(xps)
    #             ax.plot(xc, yc, color=colors[i], label=self.names[m])
    #             # ax.plot(xcs, ychat, color=colors[i], linestyle='dotted', label='%s: cs' % self.names[m])
    #             ax2.plot(xp, yp, color=colors[i])
    #             # ax2.plot(xps, yphat, color=colors[i], linestyle='dotted')
    #             i += 1
    #     if plot:
    #         ax.set_title('%s Delta Curve' % datetime.strftime(self.date, '%Y-%m-%d'))
    #         ax.invert_xaxis()
    #         ax2.invert_xaxis()
    #         ax.set_xlabel('Delta')
    #         ax.set_ylabel('IV')
    #         ax.legend(loc='lower right')
    #         plt.show()

    def get_ivs(self, *kargs, method='fit'):
        for m in self.maturities:
            try:
                if method == 'fit':
                    self.atm_ivs[self.names[m]] = float(self.delta_curves[m][0.5])
                    delta_call_ivs = {}
                    delta_put_ivs = {}
                    for delta in kargs:
                        delta_call_ivs[delta] = float(self.delta_curves[m][1 - delta])
                        delta_put_ivs[delta] = float(self.delta_curves[m][delta])  # CAUTION: 注意此处应为-delta
                    self.delta_call_ivs[self.names[m]] = delta_call_ivs
                    self.delta_put_ivs[self.names[m]] = delta_put_ivs
                # else:
                #     self.atm_ivs[self.names[m]] = self.strike_curves[m][self.atm_ks[m]]
                #     delta_call_ivs = {}
                #     delta_put_ivs = {}
                #     for delta in kargs:
                #         # 寻找距delta最近的call, put iv
                #         delta_call_ivs[delta] = self.call_wings[m].values[
                #             np.argmin(np.abs(np.array(self.call_wings[m].keys()) - delta))]
                #         delta_put_ivs[delta] = self.put_wings[m].values[
                #             np.argmin(np.abs(np.array(self.put_wings[m].keys()) + delta))]  # CAUTION: 注意此处应为 - -delta = + delta
                #     self.delta_call_ivs[self.names[m]] = delta_call_ivs
                #     self.delta_put_ivs[self.names[m]] = delta_put_ivs
            except TypeError:
                self.atm_ivs[self.names[m]], self.delta_call_ivs[self.names[m]], \
                    self.delta_put_ivs[self.names[m]] = None, None, None

    def get_skewness(self):
        for m in self.maturities:
            try:
                skew = {}
                skew['both'] = (self.delta_put_ivs[self.names[m]][0.25] -
                                self.delta_call_ivs[self.names[m]][0.25]) / self.atm_ivs[self.names[m]]
                skew['left'] = (self.delta_put_ivs[self.names[m]][0.25] -
                                self.atm_ivs[self.names[m]]) / self.atm_ivs[self.names[m]]
                skew['right'] = (self.delta_call_ivs[self.names[m]][0.25] -
                                 self.atm_ivs[self.names[m]]) / self.atm_ivs[self.names[m]]
                self.skew[self.names[m]] = skew
            except TypeError:
                self.skew[self.names[m]] = None

    def curve_to_df(self, curves, maturity):
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[self.names[maturity]], ['K', 'IV', 'Delta']])
        )
        df[(self.names[maturity], 'K')] = list(curves[maturity].keys())
        df[(self.names[maturity], 'IV')] = curves[maturity].values
        df[(self.names[maturity], 'Delta')] = self.greeks[maturity].drop_duplicates(
            subset='K', keep='first')['Delta'].values
        return df

    def curves_to_df(self):
        df = pd.DataFrame()
        for m in self.maturities:
            df = pd.concat([df, self.curve_to_df(self.strike_curves, m)], axis=1)
        return df
