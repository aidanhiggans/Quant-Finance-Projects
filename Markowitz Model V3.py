import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import scipy.optimize as optimization

# Number of different portfolios generated with random weights
NUM_PORTFOLIOS = 700000

# Rate of return of risk-free security (such as a treasury bill or bond) which will be used to calculate Sharpe Ratio
RISK_FREE_RATE = 0.03

# Benchmark ETF or Stock for comparison
BENCHMARK = 'SPY'

# Cap weight for an individual stock (i.e. the maximum amount that can be invested in a single stock)
CAP = 0.2

# Stocks to be included in the Portfolio
stocks = ['AAPL', 'WMT', 'TSLA','AMZN', 'GE', 'DB', 'MSFT' , 'UNH', 'JPM', 'BRK-B','XOM','JNJ']


# Historical Data - Defining the START and END Dates
start_date = '2015-01-01'
end_date = '2025-01-01'

# -----------------------------------
# DOWNLOADING DATA FROM YAHOO FINANCE
# -----------------------------------

def download_data():
    # Dictionary to store stock data - Stock Name as Key and Stock Price as Value
    stock_data = {}

    for stock in stocks:
        # Considering Closing Prices
        ticker = yf.Ticker(stock)
        stock_data[stock] = ticker.history(start=start_date, end=end_date)['Close']

    return pd.DataFrame(stock_data)

# ----------------------------
# PLOTTING STOCK PRICE HISTORY
# ----------------------------

def show_data(data):
    data.plot(figsize=(10, 6))
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.title('Stock Price History')
    plt.show()

# -----------------------------
# CALCULATING LOG DAILY RETURNS
# -----------------------------

def calculate_return(data):
    # Log daily returns are used through the following formula: ln(S(t)/S(t-1))
    # To capture the formula data.shift(1) function is used where data values are moved one step to the right
    # for e.g. data = 1 2 3 4 5 ...
    # data.shift(1) =   1 2 3 4 5 ...
    # since first value is divided by an undefined number, the output shall start from second row (index 1)
    # Log returns are used for normalization purposes
    # This is done so that stocks that differ greatly in stock price can still be compared accurately
    log_return = np.log(data/data.shift(1))
    return log_return.dropna()

# --------------------------------------
# CALCULATING MEAN AND COVARIANCE MATRIX
# --------------------------------------

def show_statistics(returns):
    # Mean of returns is multiplied by number of trading days in a year (252) to obtain annual return metric
    print(returns.mean() * 252)
    # Obtaining the covariance
    print(returns.cov() * 252)

# ------------------------------------------------------------------------------
# CALCULATING ANNUAL RETURN (MEAN) AND PORTFOLIO VOLATILITY (STANDARD DEVIATION)
# ------------------------------------------------------------------------------

def show_mean_and_variance(returns, weights):
    portfolio_log_return = np.sum(returns.mean() * weights * 252)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    # Converting log returns to arithmetic returns
    # There is no need to convert volatilit from log returns to volatility from true arithmetic returns, as the difference is negligible
    portfolio_arithmetic_return = np.exp(portfolio_log_return) -1

    print("Expected Portfolio Return:", portfolio_arithmetic_return)
    print("Expected Portfolio Volatility:", portfolio_volatility)

# ----------------------------------------------------------------------
# GENERATING PORTFOLIOS WITH RANDOM WEIGHTS USING MONTE-CARLO SIMULATION
# ---------------------------------------------------------------------

def generate_portfolio(returns):

    portfolio_weights = []
    portfolio_means = []
    portfolio_risks = []

    # Underscore in the for loop is used to avoid using an index for the iteration
    for _ in range(NUM_PORTFOLIOS):

        # An array of random numbers (uniformly between 0 and 1) is created with the same length as the number of stocks
        w = np.random.random(len(stocks))
        # Weights are to be normalized so that the sum of the weights is equal to 1
        # For e.g. if w = [0.6, 0.3, 0.9] then sum(w) = 1.8
        # Each weight is now divided by 1.8
        # Now normalized weights are [0.333, 0.5, 0.167] and the sum is equal to 1
        w /= np.sum(w)
        portfolio_weights.append(w)
        # Converting log returns to arithmetic returns for output
        port_log_return = np.sum(returns.mean() * w) * 252
        portfolio_means.append(np.exp(port_log_return) - 1)
        portfolio_risks.append(np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w))))

    return np.array(portfolio_weights), np.array(portfolio_means), np.array(portfolio_risks)

# ------------------------------------------------
# PLOTTING THE RANDOM PORTFOLIOS AS A SCATTER PLOT
# ------------------------------------------------

def show_portfolios(volatilities, returns):
    colors = [(0, "midnightblue"),(0.1, "darkblue"),(0.2, "blue"),(0.4, "cyan"),(0.45, "aquamarine"),(0.5, "lime"),(0.55, "yellow"),(0.6, "orange"),(0.8, "red"),(0.9, "darkred"),(1, "maroon")]
    cmap = LinearSegmentedColormap.from_list("colors", colors)
    plt.figure(figsize=(10, 6))
    plt.scatter(volatilities, returns, c = returns/volatilities, cmap = cmap, marker = 'o', edgecolor = 'black')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label = 'Sharpe Ratio')
    plt.show()

# -----------------------------------------------
# CALCULATING RETURN, VOLATILITY AND SHARPE RATIO
# -----------------------------------------------

def statistics(weights, returns):
    portfolio_log_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T,np.dot(returns.cov() * 252, weights)))
    # Converting log returns to arithmetic returns for final output
    # This is done by using the formula Arithmetic_Returns = e^(Log_Returns) - 1
    portfolio_arithmetic_return = np.exp(portfolio_log_return) - 1
    # Sharpe Ratio is calculated as Portfolio Return - Rate of return of Risk-free Security/ Standard Deviation
    sharpe_ratio = (portfolio_arithmetic_return - RISK_FREE_RATE)/ portfolio_volatility
    return np.array([portfolio_arithmetic_return, portfolio_volatility, sharpe_ratio])

# --------------------------------
# OBTAINING DATA FOR BENCHMARK ETF
# --------------------------------

def download_data_benchmark():
    # Dictionary to store benchmark data - Stock Name as Key and Stock Price as Value
    benchmark_data = {}
    # Considering Closing Prices
    ticker = yf.Ticker(BENCHMARK)
    benchmark_data[BENCHMARK] = ticker.history(start=start_date, end=end_date)['Close']
    return pd.DataFrame(benchmark_data)

# --------------------------------------
# OBTAINING STATISTICS FOR BENCHMARK ETF
# --------------------------------------

def benchmark_statistics(benchmark_prices):
    # Computing log daily returns for benchmark ETF
    bench_log_returns = np.log(benchmark_prices / benchmark_prices.shift(1)).dropna()
    # Annual mean log return and volatility
    bench_log_return = bench_log_returns.mean() * 252
    bench_volatility = np.sqrt(bench_log_returns.var() * 252)
    # Converting log returns to arithmetic mean
    bench_arithmetic_return = np.exp(bench_log_return) - 1
    # Computing Sharpe Ratio for benchmark ETF
    bench_sharpe_ratio = (bench_arithmetic_return - RISK_FREE_RATE) / bench_volatility
    return np.array([bench_arithmetic_return, bench_volatility, bench_sharpe_ratio])

# -----------------------------------------------------
# CREATING MINIMIZATION FUNCTION (MAXIMUM SHARPE RATIO)
# -----------------------------------------------------

def min_function_sharpe(weights, returns):
    # The scipy optimize module can be used to find the minimum of a function
    # Goal is to obtain portfolio with maximum Sharpe Ratio (index 2 in the statistics function)
    # To find portfolio with maximum Sharpe Ratio, the function shall be negated
    # Scipy optimize module will then be used to find the minimum of this negated function, as min -f(x) = max f(x)
    return -statistics(weights, returns)[2]

# -----------------------------
# FINDING THE OPTIMAL PORTFOLIO
# -----------------------------

def optimize_portfolio(weights, returns):
    # Constraints are defined as a dictionary with type being the key and eq (equation) being the value
    # Another key-value pair is defined, with fun (function) being the key and lambda expression being the value
    # Lambda expression is used to ensure that sum of the weights is equal to 1 (one of the two main constraints for this Markowitz Model)
    # Since np.sum(x) is equal to 1 (sum of weights is equal to 1) then np.sum(x) - 1 is equal to 0
    # Scipy optimization function deals with functions equal to 0, which is why it is necessary for the lambda expression (np.sum(x) - 1) is equal to 0
    # Second constraint is that maximum weight is 1, i.e. when 100% of the money is invested in a single stock
    # Bounds are created through tuples and a for loop that generate numbers between 0 and 1 as the weight for each stock
    # Function to be optimized is min_function_sharpe, with the first weight being the initial value, and returns is the argument
    # Optimization method to be used is SLSQP
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, CAP) for _ in range(len(stocks)))
    return optimization.minimize(fun=min_function_sharpe, x0=weights[0], args=returns, method='SLSQP', bounds=bounds, constraints=constraints)

# ---------------------------------------------------
# CREATING MINIMIZATION FUNCTION (MINIMUM VOLATILITY)
# ---------------------------------------------------

def min_function_volatility(weights, returns):
    # This function will be used to find the portfolio with the minimum volatility
    # Index 1 is used in order to return volatility only
    return statistics(weights, returns)[1]

# ---------------------------------------------
# FINDING THE PORTFOLIO WITH MINIMUM VOLATILITY
# ---------------------------------------------

def optimize_min_variance(weights, returns):
    # The same optimization method is used, only difference is the function used
    constraints = {'type': 'eq', 'fun' : lambda x : np.sum(x) -1}
    bounds = tuple((0,CAP) for _ in range(len(stocks)))
    return optimization.minimize(fun = min_function_volatility, x0 = weights[0], args = returns, method = 'SLSQP', bounds = bounds, constraints = constraints)

# -------------------------------------
# PRINTING DETAILS OF OPTIMAL PORTFOLIO
# -------------------------------------

def print_details(optimum, min_var, returns, benchmark_prices):
    # Optimal portfolio details
    # Solution will be contained in the array x
    print("Optimal Portfolio:", optimum['x'].round(2))
    print("Expected Optimal Portfolio Return, Volatility and Sharpe Ratio:", statistics(optimum['x'].round(2), returns))

    # Minimum variance portfolio details
    # Solution will be contained in the array x
    print("Minimum Variance Portfolio:", min_var['x'].round(2))
    print("Expected Minimum Variance Portfolio Return, Volatility and Sharpe Ratio:", statistics(min_var['x'], returns))

    # Benchmark ETF details
    bench_stats = benchmark_statistics(benchmark_prices[BENCHMARK])
    print(f"Benchmark Return, Volatility and Sharpe Ratio ({BENCHMARK}):", bench_stats)

# ------------------------------------------
# PLOTTING OPTIMAL PORTFOLIO ON SCATTER PLOT
# ------------------------------------------

def show_portfolio_plot(opt, min_var, rets, portfolio_vols, portfolio_rets, benchmark_prices):
    colors = [(0, "midnightblue"),(0.1, "darkblue"),(0.2, "blue"),(0.4, "cyan"),(0.45, "aquamarine"),(0.5, "lime"),(0.55, "yellow"),(0.6, "orange"),(0.8, "red"),(0.9, "darkred"),(1, "maroon")]
    cmap = LinearSegmentedColormap.from_list("colors", colors)
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_vols, portfolio_rets, c = portfolio_rets/portfolio_vols, cmap = cmap, marker = 'o', edgecolor = 'black')
    plt.grid(True)
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.colorbar(label = 'Sharpe Ratio')

    # Plotting optimal portfolio symbol
    opt_stats = statistics(opt['x'], rets)
    plt.plot(opt_stats[1], opt_stats[0], 'g*', markersize = 20, label = "Optimal Portfolio")

    # Plotting minimum variance portfolio symbol
    min_var_stats = statistics(min_var['x'], rets)
    plt.plot(min_var_stats[1], min_var_stats[0], 'mv', markersize = 12, label = "Minimum Variance Portfolio")

    # plotting benchmark ETF symbol
    bench_stats = benchmark_statistics(benchmark_prices[BENCHMARK])
    plt.plot(bench_stats[1], bench_stats[0], 'r^', markersize = 12, label = f"Benchmark ({BENCHMARK})")

    plt.legend()
    plt.show()

# --------------
# CODE EXECUTION
# --------------

if __name__ == '__main__':

    dataset = download_data()
    show_data(dataset)

    log_daily_returns = calculate_return(dataset)

    pweights, means, risks = generate_portfolio(log_daily_returns)
    show_portfolios(risks, means)

    optimum = optimize_portfolio(pweights, log_daily_returns)

    min_var = optimize_min_variance(pweights, log_daily_returns)

    benchmark_prices = download_data_benchmark()

    print_details(optimum, min_var, log_daily_returns, benchmark_prices)

    show_portfolio_plot(optimum, min_var, log_daily_returns, risks, means, benchmark_prices)
