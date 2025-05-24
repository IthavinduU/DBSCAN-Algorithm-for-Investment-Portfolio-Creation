# Full Combined Script: Stock Portfolio Optimization and Backtest vs SPY

import yfinance as yf
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns

# ------------------------------
# Helper Function: Portfolio Optimization
# ------------------------------

def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices, frequency=252)
    cov = risk_models.sample_cov(prices=prices, frequency=252)
    ef = EfficientFrontier(expected_returns=returns, cov_matrix=cov, weight_bounds=(lower_bound, 0.1), solver='SCS')
    weights = ef.max_sharpe()
    return ef.clean_weights()

# ------------------------------
# Placeholder: Replace with your actual data loading logic
# ------------------------------

# Example dummy `data` and `fixed_dates`
# Replace this with your actual loading logic
data = pd.DataFrame({
    'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'date': pd.date_range(start='2021-01-01', periods=100),
})
data.set_index(['date', 'ticker'], inplace=True)

# Simulate fixed_dates: mapping of start_date to selected stock tickers
fixed_dates = {
    '2022-01-01': ['AAPL', 'MSFT'],
    '2022-02-01': ['GOOGL', 'AMZN'],
    '2022-03-01': ['AAPL', 'META'],
}

# ------------------------------
# Download Prices
# ------------------------------
stocks = data.index.get_level_values('ticker').unique().tolist()
start_date_data = data.index.get_level_values('date').min() - pd.DateOffset(months=12)
end_date_data = data.index.get_level_values('date').max()

new_df = yf.download(tickers=stocks, start=start_date_data, end=end_date_data)

# ------------------------------
# Log Returns
# ------------------------------
returns_dataframe = np.log(new_df['Adj Close']).diff()
portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    try:
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
        cols = fixed_dates[start_date]

        optimization_start = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        optimization_df = new_df[optimization_start:optimization_end]['Adj Close'][cols]

        if optimization_df.empty:
            print(f"No data for optimization range {optimization_start} to {optimization_end}. Skipping {start_date}.")
            continue

        try:
            weights = optimize_weights(optimization_df, lower_bound=round(1 / (2 * len(cols)), 3))
            weights = pd.DataFrame({'ticker': list(weights.keys()), 'weight': list(weights.values())}).set_index('ticker')
        except Exception as e:
            print(f"Optimization failed for {start_date}. Error: {e}")
            weights = pd.DataFrame({'ticker': cols, 'weight': [1/len(cols)] * len(cols)}).set_index('ticker')

        temp_df = returns_dataframe[start_date:end_date]

        if temp_df.empty:
            print(f"No returns data for {start_date} to {end_date}. Skipping.")
            continue

        temp_df = temp_df.stack().reset_index()
        temp_df.columns = ['date', 'ticker', 'return']
        temp_df.set_index(['date', 'ticker'], inplace=True)

        merged_df = temp_df.join(weights, on='ticker')
        merged_df['weighted_return'] = merged_df['return'] * merged_df['weight']

        strategy_returns = merged_df.groupby(level=0)['weighted_return'].sum().to_frame('Strategy Return')
        portfolio_df = pd.concat([portfolio_df, strategy_returns], axis=0)

    except Exception as e:
        print(f"Error processing {start_date}: {e}")

portfolio_df = portfolio_df.drop_duplicates()

# ------------------------------
# Plot Daily Strategy Returns
# ------------------------------
portfolio_df['Strategy Return'].plot(kind='line', figsize=(10, 4), title='Daily Strategy Return')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# ------------------------------
# Histogram of Daily Returns
# ------------------------------
portfolio_df['Strategy Return'].plot(kind='hist', bins=20, title='Distribution of Strategy Returns')
plt.gca().spines[['top', 'right']].set_visible(False)
plt.show()

# ------------------------------
# Benchmark Comparison (SPY)
# ------------------------------
spy = yf.download(tickers='SPY', start='2015-01-01', end=dt.date.today())
spy_ret = np.log(spy[['Adj Close']]).diff().dropna().rename({'Adj Close': 'SPY Buy&Hold'}, axis=1)

# Merge SPY returns
portfolio_df = portfolio_df.merge(spy_ret, left_index=True, right_index=True, how='inner')

# ------------------------------
# Cumulative Returns Comparison
# ------------------------------
portfolio_df['Cumulative Return'] = (1 + portfolio_df['Strategy Return']).cumprod() - 1
portfolio_df['SPY Cumulative'] = (1 + portfolio_df['SPY Buy&Hold']).cumprod() - 1

plt.figure(figsize=(12, 6))
plt.plot(portfolio_df['Cumulative Return'], label='Strategy Cumulative Return')
plt.plot(portfolio_df['SPY Cumulative'], label='SPY Cumulative Return')
plt.title('Cumulative Returns: Strategy vs SPY')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# ------------------------------
# Evaluation Metrics
# ------------------------------
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    excess = returns - risk_free_rate / 252
    return excess.mean() / excess.std()

def calculate_max_drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns / running_max) - 1
    return drawdown.min()

annualized_return = (1 + portfolio_df['Strategy Return'].mean()) ** 252 - 1
annualized_vol = portfolio_df['Strategy Return'].std() * np.sqrt(252)
sharpe = calculate_sharpe_ratio(portfolio_df['Strategy Return'])
max_dd = calculate_max_drawdown(portfolio_df['Cumulative Return'])

print(f"Annualized Return: {annualized_return:.4f}")
print(f"Annualized Volatility: {annualized_vol:.4f}")
print(f"Sharpe Ratio: {sharpe:.4f}")
print(f"Max Drawdown: {max_dd:.2%}")

# ------------------------------
# Drawdown Plot
# ------------------------------
drawdown = (portfolio_df['Cumulative Return'] / portfolio_df['Cumulative Return'].cummax()) - 1

plt.figure(figsize=(12, 6))
plt.plot(drawdown, color='red')
plt.title('Portfolio Drawdown Over Time')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.grid(True)
plt.show()

# ------------------------------
# Final Summary
# ------------------------------
print(f"\nPerformance Summary:")
print(f"Portfolio Cumulative Return: {portfolio_df['Cumulative Return'].iloc[-1]:.2%}")
