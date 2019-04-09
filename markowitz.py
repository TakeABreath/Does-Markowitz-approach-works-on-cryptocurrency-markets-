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
beginDate = '2018-05-29'
endDate = '2018-11-29'
top100Currencies = allCurrencies[rankingDate].query('ranknow <= 4')
top100CurrenciesNames = list(top100Currencies["name"])

allCurrenciesFilteredByDate = allCurrencies[beginDate:endDate]
map = reduce( or_, (allCurrenciesFilteredByDate.name==i for i in top100CurrenciesNames))
top100FilteredByDate = allCurrenciesFilteredByDate[map]
del map, allCurrenciesFilteredByDate

# reorganise data pulled by setting date as index with
# columns of tickers and their corresponding adjusted prices
top100SymbolAndCloseFilteredByDate = top100FilteredByDate[["name","close"]]
markowitzDataset = top100SymbolAndCloseFilteredByDate.pivot(columns='name')
del top100FilteredByDate, top100SymbolAndCloseFilteredByDate

# calculate daily and annual returns of the stocks
returns_daily = markowitzDataset.pct_change()
returns_annual = returns_daily.mean() * 366

# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 366

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(top100Currencies)
num_portfolios = 50000

# set seed to get the same results every time
# np.random.seed(123)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,name in enumerate(top100CurrenciesNames):
    portfolio[name+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility'] + [stock+' Weight' for stock in top100CurrenciesNames]

# reorder dataframe columns
df = df[column_order]

# plot the efficient frontier with a scatter plot
plt.style.use('seaborn')
df.plot.scatter(x='Volatility', y='Returns', figsize=(10, 8), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()




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
num_assets = len(top100CurrenciesNames)
num_portfolios = 50000

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
for counter,symbol in enumerate(top100CurrenciesNames):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in top100CurrenciesNames]

# reorder dataframe columns
df = df[column_order]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
#df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
#                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
fig, ax = plt.subplots()
df.plot(kind="scatter", x='Volatility', y='Returns', c='Sharpe Ratio', cmap='RdYlGn', ax=ax);
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()





#### 3rd approach
# based on blog post here: http://blog.quantopian.com/markowitz-portfolio-optimization-2/

import numpy as np
import matplotlib.pyplot as plt
import cvxopt as opt
from cvxopt import blas, solvers
import pandas as pd

import mpld3
from mpld3 import plugins

np.random.seed(123)

# Turn off progress printing 
solvers.options['show_progress'] = False

## NUMBER OF ASSETS
n_assets = 4

## NUMBER OF OBSERVATIONS
n_obs = 1000  #original 1000

#results in a n_assets x n_obs vector, with a return for each asset in each observed period
return_vec = np.random.randn(n_assets, n_obs) 


## Additional code demonstrating the formation of a Markowitz Bullet from random portfolios:

def rand_weights(n):
    ''' Produces n random weights that sum to 1 '''
    k = np.random.rand(n)
    return k / sum(k)


def random_portfolio(returns):
    ''' 
    Returns the mean and standard deviation of returns for a random portfolio
    '''

    p = np.asmatrix(np.mean(returns, axis=1))
    w = np.asmatrix(rand_weights(returns.shape[0]))
    C = np.asmatrix(np.cov(returns))
    
    mu = w * p.T
    sigma = np.sqrt(w * C * w.T)
    
    # This recursion reduces outliers to keep plots pretty
    if sigma > 2:
        return random_portfolio(returns)
    return mu, sigma

## Uncomment to see visualization
n_portfolios = 500
means, stds = np.column_stack([
    random_portfolio(return_vec) 
    for _ in xrange(n_portfolios)
])

plt.plot(stds, means, 'o', markersize=5)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')
plt.show()



def convert_portfolios(portfolios):
    ''' Takes in a cvxopt matrix of portfolios, returns list '''
    port_list = []
    for portfolio in portfolios:
        temp = np.array(portfolio).T
        port_list.append(temp[0].tolist())
        
    return port_list


def optimal_portfolio(returns):
    ''' returns an optimal portfolio given a matrix of returns '''
    n = len(returns)
    #print n  # n=4, number of assets
    returns = np.asmatrix(returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))  #S is the covariance matrix. diagonal is the variance of each stock

    
    pbar = opt.matrix(np.mean(returns, axis=1))
    #print "pbar:", pbar

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]

    port_list = convert_portfolios(portfolios)
 
   
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]  #Different than input returns
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] #np.sqrt returns the stdev, not variance
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x'] #Is this the tangency portfolio? X1 = slope from origin?  
    #print "wt, optimal portfolio:", wt
    return np.asarray(wt), returns, risks, port_list



def covmean_portfolio(covariances, mean_returns):
    ''' returns an optimal portfolio given a covariance matrix and matrix of mean returns '''
    n = len(mean_returns)
    
    N = 100
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    S = opt.matrix(covariances)  # how to convert array to matrix?  

    pbar = opt.matrix(mean_returns)  # how to convert array to matrix?

    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    port_list = convert_portfolios(portfolios)
    
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    frontier_returns = [blas.dot(pbar, x) for x in portfolios]  
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios] 
    
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(frontier_returns, risks, 2)
    #print m1 # result: [ 159.38531535   -3.32476303    0.4910851 ]
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']  

    return np.asarray(wt), frontier_returns, risks, port_list


## Example Input from Estimates

covariances = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] # inner lists represent columns, diagonal is variance
 

mean_returns = [1.5,3.0,5.0,2.5] # Returns in DALYs

weights, returns, risks, portfolios = covmean_portfolio(covariances, mean_returns)

weights, returns, risks, portfolios = optimal_portfolio(returns_daily)
#plt.plot(stds, means, 'o') #if you uncomment, need to run 500 porfolio random simulation above

## Matplotlib Visualization:

plt.ylabel('mean')
plt.xlabel('std')
plt.plot(risks, returns, 'y-o') #risks and returns are just arrays of points along the frontier
plt.show()


## Optional interactive plot using mpld3:
## Source: http://mpld3.github.io/examples/html_tooltips.html

# fig, ax = plt.subplots()
# ax.grid(True, alpha=0.3)
#
# labels = []
# for i in range(len(risks)):
#     label = " Risk: " + str(risks[i]) + " Return: " + str(returns[i]) + " Portfolio Weights: " + str(portfolios[i])
#     labels.append(str(label))
#
# points = ax.plot(risks, returns, 'o', color='b',
#                  mec='k', ms=15, mew=1, alpha=.6)
#
# ax.set_xlabel('standard deviation')
# ax.set_ylabel('return')
# ax.set_title('Efficient Frontier', size=20)
#
# tooltip = plugins.PointHTMLTooltip(points[0], labels,
#                                    voffset=10, hoffset=10)
# plugins.connect(fig, tooltip)
#
# mpld3.show()