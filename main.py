from utils.IVCurve import *
from utils.FindContracts import *
import pandas as pd
from WindPy import w
from tqdm import tqdm
w.start()


if __name__ == '__main__':

    contract_info = pd.read_excel('./data/Opt_ID.xlsx', sheet_name='合约代码', usecols='A:J', index_col=0)
    close_info = pd.read_excel('./data/Opt_ID.xlsx', sheet_name='收盘价', usecols='A:C', index_col=0, skiprows=3)

    # ================================ 参数调整 ================================ #
    start_date = '2019-01-01'
    end_date = '2019-01-10'
    dates = [d for d in w.tdays(start_date, end_date, '').Data[0]]

    tt = 0                            # 剩余交易天数 < tt: 剔除该月份合约
    # ======================== 以下参数如不设置则为默认值 ======================== #
    kwargs = {
        'underlying': '510050.SH',  # 标的, 默认50ETF: '510050.SH'
        'n1': 6,                    # S在3以下取上下n1档, 默认6
        'n2': 5,                    # S在5以下取上下n2档, 默认5
        'n3': 4,                    # S在5以上取上下n3档, 默认4
        'synfutures_n': 2,          # 计算合成期货时使用上下synfutures_n档, 默认2
        'method': 'black',             # 'bs' or 'black', 默认'bs'
        'r_bs': 0.03,               # 默认0.03
        'r_black': 0                # 默认0
    }
    # ========================================================================= #

    cols = pd.MultiIndex.from_product([['0M', '1M', '2M', '3M'], ['both', 'left', 'right']])
    summary = pd.DataFrame(columns=cols)
    bar = tqdm(dates)
    maturities = None
    for date in bar:
        bar.set_description("Processing %s" % datetime.strftime(date, '%Y-%m-%d %H:%M:%S'))
        if maturities is None or date >= maturities[0]:
            fc = FindContracts(date, close_info, contract_info)
            maturities = fc.locate_4m()
            # fc.filter_maturities(tt)  # 调用：剩余交易天数 < tt: 剔除该月份合约
        try:
            ivc = IVCurve(date, maturities, close_info, contract_info, **kwargs)
        except:
            ivc = IVCurve(date, maturities, close_info, contract_info)

        ivc.get_strike_curves_and_greeks()  # 调用：生成strike curve, greeks
        ivc.get_delta_curves(plot=False)
        # ivc.get_fitted_strike_curves(plot=True)  # 调用：cubic spline拟合strike curve
        # ivc.get_delta_curves(interval=0.05)  # 调用：生成delta curve
        # ivc.get_fitted_delta_curves(plot=True)  # 调用：cubic spline拟合delta curve
        ivc.get_ivs(0.25, method='fit')  # method='fit' 直接填写参数Delta1, Delta2, ..., 获取atm iv和delta call, put iv
        ivc.get_skewness()
        # print('2M-1M ATM: %.2f%%' % (100*(ivc.atm_ivs['2M'] - ivc.atm_ivs['1M'])))
        # print('2M-1M 0.25Delta Call: %.2f%%' % (100*(ivc.delta_call_ivs['2M'][0.25] - ivc.delta_call_ivs['1M'][0.25])))
        # print('2M-1M 0.25Delta Put: %.2f%%' % (100*(ivc.delta_put_ivs['2M'][0.25] - ivc.delta_put_ivs['1M'][0.25])))
        # df = ivc.curves_to_df()
        append_df = pd.DataFrame(columns=cols, index=[date])
        for m in ivc.maturities:
            for k in ['both', 'left', 'right']:
                append_df[(ivc.names[m], k)] = ivc.skew[ivc.names[m]][k]
        summary = summary.append(append_df)
        summary.to_excel('./data/summary.xlsx')
