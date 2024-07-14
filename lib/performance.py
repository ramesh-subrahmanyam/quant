import pandas as pd
import numpy as np

def annualized_sharpe_ratio(pnl_series, periods_per_year=252):
    """Calculate the annualized Sharpe ratio."""
    mean_daily_return = pnl_series.mean()
    std_daily_return = pnl_series.std()
    sharpe_ratio = mean_daily_return / std_daily_return
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(periods_per_year)
    return annualized_sharpe_ratio

def compute_performance(df):
    """
    Calculate performance metrics for unslipped and slipped PnL.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'pnl', 'slipped pnl', and 'position', indexed by dates.
    
    Returns:
    pd.DataFrame: DataFrame with performance metrics.
    """
    metrics = {
        'Sharpe': [],
        '#Trades': [],
        'AvgPnL': [],
        'TotalPnL':[], 
        'Duration': [],
        'Win%': []
    }
    
    for pnl_col in ['pnl', 'slipped_pnl']:
        # Calculate Annualized Sharpe Ratio
        annualized_sharpe = annualized_sharpe_ratio(df[pnl_col])
        metrics['Sharpe'].append(annualized_sharpe)
        
        # Identify trades
        trades = []
        current_trade = []
        for date, pos in df['pos'].shift(1).fillna(0).items():
            if pos != 0:
                current_trade.append(date)
            elif pos == 0 and current_trade:
                trades.append(current_trade)
                current_trade = []
        
        # Calculate number of trades
        num_trades = len(trades)
        metrics['#Trades'].append(num_trades)
        
        # Calculate PnL per trade and trade lengths
        pnl_per_trade = []
        trade_lengths = []
        wins = 0
        
        for trade in trades:
            trade_pnl = df.loc[trade, pnl_col].sum()
            pnl_per_trade.append(trade_pnl)
            trade_lengths.append(len(trade))
            if trade_pnl > 0:
                wins += 1
        
        # Calculate Average PnL per Trade
        avg_pnl_per_trade = np.mean(pnl_per_trade) if pnl_per_trade else 0
        metrics['AvgPnL'].append(int(avg_pnl_per_trade))
        
        # Calculate Total PnL 
        metrics['TotalPnL'].append(int(sum(pnl_per_trade)))
        
        # Calculate Average Length of a Trade
        avg_trade_length = np.round(np.mean(trade_lengths) if trade_lengths else 0, 2)
        metrics['Duration'].append(avg_trade_length)
        
        # Calculate Percentage of Trades that are Wins
        eps=0.01
        percentage_wins = int((wins/(len(trades)+eps)) * 100)
        metrics['Win%'].append(percentage_wins)
    
    # Create DataFrame with performance metrics
    performance_df = pd.DataFrame(metrics, index=['unslipped', 'slipped'])
    
    return performance_df


def get_yearly_slipped_performance(df):
  """
  This function takes a DataFrame `df` indexed by date and calculates 
  year-by-year performance for the slipped strategy.

  Args:
      df (pandas.DataFrame): The DataFrame containing performance data indexed by date.

  Returns:
      pandas.DataFrame: A DataFrame with yearly performance for the slipped strategy.
  """

  yearly_performance = []
  # Extract unique years from the DataFrame
  years = list(df.index.year.unique())

  for year in years:
    # Filter data for the current year
    year_df = df.loc[df.index.year == year]
    # Calculate performance for the slipped strategy
    perf_df=compute_performance(year_df)
    yearly_performance.append([year] + list(perf_df.loc["slipped", :]))
  # Combine results into a single DataFrame (assuming consistent column names)
  yearly_performance_df = pd.DataFrame(yearly_performance, columns=["year"] + list(perf_df.columns))
  yearly_performance_df=yearly_performance_df.set_index("year")
  return yearly_performance_df


  
