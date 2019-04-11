# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:01:14 2019

@author: mgraczyk
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.parser import parse
from functools import reduce
from operator import or_
from itertools import permutations 
from itertools import combinations_with_replacement

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Set directory
os.chdir("C:/Users/mgraczyk/Documents/STUDIA/Python/Markowitz/input")

# Import Data
CryptoMarkets = pd.read_csv('crypto-markets.csv')

# Parse dates
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
allCurrencies = pd.read_csv('../input/crypto-markets.csv', parse_dates=['date'], index_col='date', date_parser=dateparse)

# Find Top 100 Cryptocurrencies by Market Capitalization and subset quotations from specified period of time 
rankingDate = '2018-11-22'
beginDate = '2018-11-22'
endDate = '2018-11-29'
topCurrencies = allCurrencies[rankingDate].query('ranknow <= 10')
topCurrenciesNames = list(topCurrencies["name"])

allCurrenciesFilteredByDate = allCurrencies[beginDate:endDate]
map = reduce( or_, (allCurrenciesFilteredByDate.name==i for i in topCurrenciesNames))
topFilteredByDate = allCurrenciesFilteredByDate[map]
del map, allCurrenciesFilteredByDate

# reorganise data pulled by setting date as index with
# columns of tickers and their corresponding adjusted prices
topSymbolAndCloseFilteredByDate = topFilteredByDate[["name","close"]]
markowitzDataset = topSymbolAndCloseFilteredByDate.pivot(columns='name')
del topFilteredByDate, topSymbolAndCloseFilteredByDate

#### 2nd approach
# calculate daily and annual returns of the stocks
returns_daily = markowitzDataset.pct_change()
returns_annual = returns_daily.mean() * 366

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 366

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(topCurrenciesNames)
num_portfolios = 100000 # m = (k+n-1)! / ((n-1)!*k!), m - number of desired portfolios, k=10 - number of posiible weights (0.1), n=10 - number of currencies

# define all combinations not randomly
#theSmallestWeight = 0.01
#allPosiibleWeights = 100
#pppermutations = list(dict.fromkeys(combinations_with_replacement(range(0, 51),num_assets)))
#weightsCombinations = []
#for i in list(pppermutations): 
#    if sum(i)==50:
#        newWeight = [float(x)/50 for x in i]
#        weightsCombinations.append(newWeight)
#        print(newWeight) 
# populate the empty lists with each portfolios returns,risk and weights
#for weights in weightsCombinations: 
#    weights = np.asarray(weights, dtype=np.float64)    
#    weights = np.random.random(num_assets)
#    weights /= np.sum(weights)
#    returns = np.dot(weights, returns_annual)
#    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
#    sharpe = returns / volatility
#    sharpe_ratio.append(sharpe)
#    port_returns.append(returns)
#    port_volatility.append(volatility)
#    stock_weights.append(weights)

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(topCurrenciesNames):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in topCurrenciesNames]

# reorder dataframe columns
df = df[column_order]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
#df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
#                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
fig, ax = plt.subplots()
df.plot(kind="scatter", x='Volatility', y='Returns', c='Sharpe Ratio', cmap='RdYlGn', ax=ax);
df.plot(1, 15, "or")
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

