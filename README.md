# Markowitz Model - Modern Portfolio Theory

## Project Overview
This project implements the **Markowitz Model - one of the fundamental models of Modern Portfolio Theory**, using Python to construct,simulate and optimize investment portfolios 

The model uses **Monte Carlo simulations** to generate thousands of random portfolios, evaluates their performance (expected return, volatility, and Sharpe ratio), and identifies:
- The **optimal portfolio** (maximizing Sharpe ratio)  
- The **minimum variance portfolio**  
- Comparison against a **benchmark ETF (such as S&P 500 – SPY)**

The implementation includes flexible parameters for seamless sensitivity analysis, such as the number of simulated portfolios, risk-free rate, benchmark selection, allocation caps, customizable asset universe, and investment timeline

The project demonstrates key concepts in **quantitative finance, portfolio theory, and optimization**.

---

## Features
- Download historical price data from Yahoo Finance (`yfinance`)  
- Compute **log returns, expected returns, volatility, covariance**  
- Run **Monte Carlo simulation** to generate random portfolios  
- Calculate and plot the **Efficient Frontier**  
- Optimize portfolio allocation using `scipy.optimize`:  
  - **Maximum Sharpe ratio portfolio**  
  - **Minimum volatility portfolio**  
- Benchmark portfolio performance against the **S&P 500 ETF (SPY)**  
- Visualize portfolio results with **risk-return scatter plots**  

---

## Methodology
1. **Data Collection:** Download stock price history from Yahoo Finance.  
2. **Visualizing Stock Price History:** Plotting stock prices over the specified time interval.
3. **Return Calculation:** Compute log daily returns, annualize returns and covariance.  
4. **Monte Carlo Simulation:** Generate random portfolios under a weight constraint.
6. **Portfolio Statistics:** Calculate expected return, volatility, and Sharpe ratio.  
7. **Optimization: Sequential Least Squares Programming (SLSQP)** to find maximum Sharpe Ratio and minimum volatility.  
8. **Visualization:** The classic Volatility-Return Scatter Plot is used to visualize the random portfolios and efficient frontier, as well as the Optimal, minimum variance and benchmark portfolios 

---

## Tech Stack
- **Language:** Python 3  
- **Libraries:**  
  - `numpy` (numerical computations)
  - `yfinance` (financial data)
  - `pandas` (data manipulation)  
  - `matplotlib` (visualization)    
  - `scipy.optimize` (portfolio optimization)  

---

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/portfolio-optimization.git
   cd portfolio-optimization
