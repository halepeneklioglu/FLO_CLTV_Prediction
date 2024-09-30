import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("datasets/flo_data_20k.csv")

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = (quartile3 + 1.5 * interquantile_range)
    low_limit = (quartile1 - 1.5 * interquantile_range)
    return round(low_limit), round(up_limit)

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"])

col_names = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline", "customer_value_total_ever_online"]

for col in col_names:
    replace_with_thresholds(df, col)

df["total_order_num"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

from datetime import datetime
columns_to_convert = df[["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]]

for col in columns_to_convert:
    df[col] = pd.to_datetime(df[col])

df["last_order_date"].max()
today_date = dt.datetime(2021, 6, 1)

df["date"] = (df["last_order_date"] - df["first_order_date"]).dt.days

cltv = df.groupby("master_id").agg({"date": lambda date: date,
                                    "first_order_date": lambda first_order_date: (today_date - first_order_date).dt.days,
                                    "total_order_num": lambda total_order_num: total_order_num,
                                    "total_customer_value": lambda total_customer_value: total_customer_value})

cltv["total_customer_value"] = cltv["total_customer_value"] / cltv["total_order_num"]

cltv = cltv.reset_index()

cltv.columns = ["customer_id", "recency_cltv_weekly", "T_weekly", "frequency", "monetary_cltv_avg"]

cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"] / 7

cltv["T_weekly"] = cltv["T_weekly"] / 7

cltv = cltv[(cltv['frequency'] > 1)]

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv['frequency'],
        cltv['recency_cltv_weekly'],
        cltv['T_weekly'])

cltv["exp_sales_3_month"] = bgf.predict(4 * 3,
                                        cltv['frequency'],
                                        cltv['recency_cltv_weekly'],
                                        cltv['T_weekly'])

cltv["exp_sales_6_month"] = bgf.predict(4 * 6,
                                        cltv['frequency'],
                                        cltv['recency_cltv_weekly'],
                                        cltv['T_weekly'])

plot_period_transactions(bgf)
plt.show()

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                                                    cltv['monetary_cltv_avg'])

cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency_cltv_weekly'],
                                   cltv['T_weekly'],
                                   cltv['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)

cltv.sort_values(by="cltv", ascending=False)[:20]

cltv["cltv_segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])

cltv.groupby("cltv_segment").agg({"exp_sales_6_month": "sum",
                                            "exp_average_value": "sum",})











