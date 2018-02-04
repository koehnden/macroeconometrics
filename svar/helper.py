"""
Script to prepare different macroeconomic time series 

@author: dkoehn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
import os

def convert_date_column(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    return df

def compute_growth_rate(df):
    return df.pct_change()[3:]*100

def convert_to_quarterly(df):
    df_quarterly = df.resample('Q').sum()
    df_quarterly.index += pd.Timedelta('1 days')
    return df_quarterly


def join_dfs_by_index(dfs):
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = pd.merge(df_merged,
                             df, 
                             how='inner', 
                             left_index=True, 
                             right_index=True)
    return df_merged


def plot_ma_mstd(df):
    ma = df.rolling(20).mean()[19:]
    mstd = df.rolling(20).std()[19:]
    plt.figure()
    df_adjusted = df[19:]
    plt.plot(df_adjusted.index, df_adjusted, 'k')
    plt.plot(ma.index, ma, 'b')
    lower = ma-2*mstd
    upper = ma+2*mstd
    plt.plot(lower.index, lower, 'b', linestyle= '--')
    plt.plot(upper.index, upper, 'b', linestyle='--')
    return plt










                    


