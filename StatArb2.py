# --------------------------------------------
# IMPORTS [CELL]
# --------------------------------------------
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
from sklearn.model_selection import ParameterGrid
from typing import List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------
# HARDCODED PARAMS [CELL, TO BE MOVED TO THE END]
# --------------------------------------------
tickers = [
    "NVDA",
    "MSFT",
    "AAPL",
    "AMZN",
    "META",
    "AVGO",
    "GOOGL",
    "TSLA",
    "GOOG",
    "BRK-B",
]
start_date = "2010-01-01"
end_date = "2019-12-31"
transaction_cost = 0.0  # pct
leverage = 1  # multiplier
cash = 500000  # starting cash for position sizing

# ---------------------------
# Data fetching & prep [CELL]
# ---------------------------


def download_data(
    stock_list: List[str], start_date: str, end_date: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates & downloads open and close price dataframes for given stock list and date range.

    Inputs:
    - stock_list (List[str]): List of stocks
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - end_date (str): End date in 'YYYY-MM-DD' format

    Returns: Tuple[open_df, `close_df`]: A tuple of two DataFrames containing open and close prices respectively.
    """
    data = yf.download(stock_list, start=start_date, end=end_date)
    # Keep only 'Open' and 'Close' prices
    data = data.loc[:, data.columns.get_level_values(0).isin(["Open", "Close"])]

    # Clean data: remove inf and NaN values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Split into two separate DataFrames
    open_df = data["Open"].astype(float)
    close_df = data["Close"].astype(float)

    return open_df, close_df


# --------------------------------------------
# Generate signals and hedge ratios [CELL]
# --------------------------------------------


def generate_signals_and_hedge_ratios(
    close_data: pd.DataFrame,
    tickers: List[str],
    entry_threshold: float = 1.0,
    exit_threshold: float = 1.0,
    coint_threshold: float = 0.01,
    window_size: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TODO: Comment u shit

    """
    # Initialize DataFrames to zeros, using ticker pairs as column names
    signals = pd.DataFrame(
        0,
        index=close_data.index,
        columns=[
            f"{ticker_a}_{ticker_b}"
            for i, ticker_a in enumerate(tickers)
            for j, ticker_b in enumerate(tickers)
            if i < j
        ],
    )
    hedge_ratios = pd.DataFrame(
        0.0,
        index=close_data.index,
        columns=[
            f"{ticker_a}_{ticker_b}"
            for i, ticker_a in enumerate(tickers)
            for j, ticker_b in enumerate(tickers)
            if i < j
        ],
    )

    # Download warmup data based on start date of close_data,and prepend to close_data
    trading_start_date = close_data.index[0]
    warmup_end_date = trading_start_date - timedelta(1)
    warmup_start_date = warmup_end_date - timedelta(window_size)

    warmup_data_open, warmup_data_close = download_data(
        tickers, warmup_start_date, warmup_end_date
    )

    # Prepend to close_data
    close_data_with_warmup = pd.concat([warmup_data_close, close_data])

    # Loop over all pairs of stocks
    for i, ticker_a in tqdm(enumerate(tickers), desc="Outer"):
        for j, ticker_b in tqdm(enumerate(tickers), desc="Inner"):
            if i >= j:  # To avoid redundant pairs (A,B) and (B,A)
                continue

            # Extract stock data for the pair (using yesterday's close for today’s position)
            data_a = close_data_with_warmup[ticker_a].shift(1)[
                1:
            ]  # Use yesterday's closing price for stock A
            data_b = close_data_with_warmup[ticker_b].shift(1)[
                1:
            ]  # Use yesterday's closing price for stock B

            # Sliding window regression and cointegration test (using all available data up to current time point)
            for t in tqdm(range(window_size, len(close_data)), desc="Time Loop"):
                # Use all data from start to time t (sliding window)
                y = data_a.iloc[
                    t - window_size : t + 1
                ]  # Dependent variable (stock A), t+1 because iloc is exclusive
                x = data_b.iloc[
                    t - window_size : t + 1
                ]  # Independent variable (stock B)

                # Skip cointegration test if there's insufficient data (e.g., t < 2)
                if len(y) <= 5 or len(x) <= 5:
                    continue  # Skip if not enough data for cointegration test

                # Perform cointegration test
                _, p_value, _ = st.coint(y, x)

                # Only proceed if the pair is cointegrated
                if p_value >= coint_threshold:
                    continue  # Skip this pair if not cointegrated

                # Add constant term to independent variable for regression
                x = sm.add_constant(x)

                # Perform linear regression to get the hedge ratio (n)
                model = sm.OLS(y, x).fit()
                hedge_ratio = model.params.iloc[
                    1
                ]  # coefficient for stock B (slope coefficient)

                # build spread series over the same expanding window
                spread_series = y - hedge_ratio * x.iloc[:, 1]

                # since you want the window to match the expanding window (0..t), you don't need rolling:
                mu = spread_series.mean()
                sd = spread_series.std(ddof=1)

                last_spread = spread_series.iloc[-1]
                z_score = 0 if sd == 0 else (last_spread - mu) / sd

                # Generate signal based on z-score and thresholds (1 for long, -1 for short, 0 for exit or no position)
                pair_name = f"{ticker_a}_{ticker_b}"
                if z_score < -entry_threshold:  # Long the spread
                    signals.loc[signals.index[t], pair_name] = (
                        1  # Long stock A (underperforming)
                    )
                elif z_score > entry_threshold:  # Short the spread
                    signals.loc[signals.index[t], pair_name] = (
                        -1
                    )  # Short stock A (underperforming)
                elif abs(z_score) < exit_threshold:  # Exit signal
                    continue
                    # signals[pair_name].iloc[t] = 0  # Exit position
                else:
                    continue
                    # signals[pair_name].iloc[t] = 0  # No action

                # Store hedge ratio for the pair
                hedge_ratios.loc[hedge_ratios.index[t], pair_name] = hedge_ratio

    hedge_ratios = hedge_ratios.apply(lambda x: abs(x))

    return signals, hedge_ratios

# --------------------------------------------



# --------------------------------------------
# CALCULATE PNL [CELL]
# --------------------------------------------
def create_positions_dict(tickers: List[str]) -> dict:
    """
    Creates a dictionary to hold positions for each ticker pair.

    {'A_B': (qtyA, qtyB), ...}
    """
    positions = {}
    # k: A_B, v: (qtyA, qtyB)
    for i, ticker_a in enumerate(tickers):
        for j, ticker_b in enumerate(tickers):
            if i < j:
                pair_name = f"{ticker_a}_{ticker_b}"
                positions[pair_name] = (0, 0)  # Initialize quantities to zero
    return positions


def calculate_position_sizes(
    total_cash: float,
    priceA: float,
    priceB: float,
    hedge_ratio: float,
    sizing_pct: float = 0.03,
    leverage: float = 1,
) -> Tuple[float, float]:
    """
    Assumes each position is 3% of free cash

    Leverage is applied to position AFTER calculating position sizes.

    Formula derived from:
    Shares_A = C / ((1+n) * Price_A)
    Shares_B = nC / ((1+n) * Price_B)

    Where C is position size and n is hedge ratio (as decimal).

    Returns: Tuple of (qtyA, qtyB). Note that qtyB is always positive here; sign is handled in main logic.
    """
    leverage = max(leverage, 1)  # Ensure leverage is at least 1
    position_size = total_cash * sizing_pct * leverage  # position size in $$ value

    qtyA = position_size / ((1 + hedge_ratio) * priceA)
    qtyB = (hedge_ratio * position_size) / ((1 + hedge_ratio) * priceB)

    return qtyA, qtyB


def calculate_transaction_costs(
    # Will always be calculated on open prices
    qtyA: float,
    qtyB: float,
    priceA: float,
    priceB: float,
    transaction_cost: float,
) -> float:
    costA = abs(qtyA * priceA) * transaction_cost
    costB = abs(qtyB * priceB) * transaction_cost
    return costA + costB


def calculate_intraday_long(
    qtyA: float, qtyB: float, openA: float, openB: float, closeA: float, closeB: float
) -> float:
    """
    Calculates intraday PnL when going long on spread. This represents a long pos in A and a short pos in B.

    Given as close - open for A and open - close for B.
    """
    pnlA = qtyA * (closeA - openA)
    pnlB = qtyB * (openB - closeB)
    return pnlA + pnlB


def calculate_intraday_short(
    qtyA: float, qtyB: float, openA: float, openB: float, closeA: float, closeB: float
) -> float:
    """
    Calculates intraday PnL when going short on spread. This represents a short pos in A and a long pos in B.

    Given as open - close for A and close - open for B.
    """
    pnlA = qtyA * (openA - closeA)
    pnlB = qtyB * (closeB - openB)
    return pnlA + pnlB


def calculate_interday_long(
    qtyA: float,
    qtyB: float,
    closeA_prev: float,
    closeB_prev: float,
    openA: float,
    openB: float,
) -> float:
    """
    Calculates interday PnL when going long on spread. This represents a long pos in A and a short pos in B.

    Given as open - prev close for A and prev close - open for B.

    """
    pnlA = qtyA * (openA - closeA_prev)
    pnlB = qtyB * (closeB_prev - openB)
    return pnlA + pnlB


def calculate_interday_short(
    qtyA: float,
    qtyB: float,
    closeA_prev: float,
    closeB_prev: float,
    openA: float,
    openB: float,
) -> float:
    """
    Calculates interday PnL when going short on spread. This represents a short pos in A and a long pos in B.

    Given as prev close - open for A and open - prev close for B.
    """
    pnlA = qtyA * (closeA_prev - openA)
    pnlB = qtyB * (openB - closeB_prev)
    return pnlA + pnlB

# ----------- HELPERS END ---------

def calculate_pnl_from_signals_and_hedge_ratios(
    signals: pd.DataFrame,
    hedge_ratios: pd.DataFrame,
    open_data: pd.DataFrame,
    close_data: pd.DataFrame,
    cash:float = 500000,
    transaction_cost_pct: float = 0.0,
    leverage: float = 1,
) -> tuple[pd.Series, pd.Series]:
    """
    Calculates daily PnL based on generated signals and hedge ratios.

    Returns daily PnL series and daily cash position series.
    """

    assert(len(signals) == len(hedge_ratios) == len(open_data) == len(close_data)), "DataFrames must have the same length"

    # init data structures
    positions = create_positions_dict(tickers)
    pnl_series = pd.Series(0.0, index=signals.index)
    cash_series = pd.Series(0.0, index=signals.index)
    cash_series.iloc[0] = cash

    # loop through each time point
    # start from start day till last day
    for t in range(len(signals)):
        current_signals:pd.Series = signals.iloc[t]
        current_hedge_ratios:pd.Series = hedge_ratios.iloc[t]
        open_prices:pd.Series = open_data.iloc[t]
        close_prices:pd.Series = close_data.iloc[t]        
        prev_signals:pd.Series = signals.iloc[t-1] if t > 0 else pd.Series(0, index=signals.columns)
        prev_close_prices:pd.Series = close_data.iloc[t-1] if t > 0 else pd.Series(0.0, index=close_data.columns)

        for pair in signals.columns:
            signal = current_signals[pair]
            hedge_ratio = current_hedge_ratios[pair]
            tickerA, tickerB = pair.split("_")
            openA = open_prices[tickerA]
            openB = open_prices[tickerB]
            closeA = close_prices[tickerA]
            closeB = close_prices[tickerB]
            prev_signal = prev_signals[pair]
            closeA_prev = prev_close_prices[tickerA]
            closeB_prev = prev_close_prices[tickerB]

            # Case 1: 0 -> 0 : Do nothing
            if prev_signal == 0 and signal == 0:
                continue
            # Case 2: 0 -> 1 : Enter long position
            elif prev_signal == 0 and signal == 1:
                # Calculate position sizes
                new_qtyA, new_qtyB = calculate_position_sizes(
                    cash, openA, openB, hedge_ratio, leverage=leverage
                )
                transaction_cost_entry = calculate_transaction_costs(new_qtyA, new_qtyB, openA, openB, transaction_cost_pct)
                purchase_cost = (openA * new_qtyA) - (openB * new_qtyB) # Cost = Buy A - Sell B. if positive, cash outflow
                cash -= transaction_cost_entry
                cash -= purchase_cost
                pnl_series.iloc[t] -= transaction_cost_entry
                pnl_series.iloc[t] += calculate_intraday_long(
                    new_qtyA, new_qtyB, openA, openB, closeA, closeB
                )
                cash_series.iloc[t] = cash # TODO: REMOVE

                positions[pair] = (new_qtyA, new_qtyB)
                print(
                    f"Entered long spread on {pair} at time {signals.index[t]}: qtyA={new_qtyA}, qtyB={new_qtyB}, openA={openA}, openB={openB}, position_cash = {(new_qtyA * openA) - (new_qtyB * openB)}, cash={cash}, hedge_ratio={hedge_ratio}, leverage={leverage}"
                )
                print(
                    f"Intraday long PnL: {calculate_intraday_long(new_qtyA, new_qtyB, openA, openB, closeA, closeB)}"
                )
            # Case 3: 0 -> -1 : Enter short position
            elif prev_signal == 0 and signal == -1:
                # Calculate position sizes
                new_qtyA, new_qtyB = calculate_position_sizes(
                    cash, openA, openB, hedge_ratio, leverage=leverage
                )
                transaction_cost_entry = calculate_transaction_costs(new_qtyA, new_qtyB, openA, openB, transaction_cost_pct)
                purchase_cost = (openB * new_qtyB) - (openA * new_qtyA) # Cost = Buy B - Sell A. if positive, cash outflow
                cash -= transaction_cost_entry
                cash -= purchase_cost
                pnl_series.iloc[t] -= transaction_cost_entry
                pnl_series.iloc[t] += calculate_intraday_short(
                    new_qtyA, new_qtyB, openA, openB, closeA, closeB
                )
                cash_series.iloc[t] = cash # TODO: REMOVE

                positions[pair] = (new_qtyA, new_qtyB)

                print(
                    f"Entered short spread on {pair} at time {signals.index[t]}: qtyA={new_qtyA}, qtyB={new_qtyB}, openA={openA}, openB={openB}, position_cash = {(new_qtyB * openB) - (new_qtyA * openA)}, cash={cash}, hedge_ratio={hedge_ratio}"
                )
                print(
                    f"Intraday short PnL: {calculate_intraday_short(new_qtyA, new_qtyB, openA, openB, closeA, closeB)}"
                )
            # Case 4: 1 -> 1 : Maintain long position
            elif prev_signal == 1 and signal == 1:
                qtyA, qtyB = positions[pair]
                pnl_series.iloc[t] += calculate_interday_long(
                    qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                )
                pnl_series.iloc[t] += calculate_intraday_long(
                    qtyA, qtyB, openA, openB, closeA, closeB
                )
                cash_series.iloc[t] = cash # TODO: REMOVE
                print(f'Maintaining long spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, cash={cash}, hedge_ratio={hedge_ratio}, leverage={leverage}')
            # case 5: -1 -> -1 : Maintain short position
            elif prev_signal == -1 and signal == -1:
                qtyA, qtyB = positions[pair]
                pnl_series.iloc[t] += calculate_interday_short(
                    qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                )
                pnl_series.iloc[t] += calculate_intraday_short(
                    qtyA, qtyB, openA, openB, closeA, closeB
                )
                cash_series.iloc[t] = cash # TODO: REMOVE
                print(f'Maintaining short spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, cash={cash}, hedge_ratio={hedge_ratio}, leverage={leverage}')
            # Case 6: 1 -> 0 : Exit long position
            elif prev_signal == 1 and signal == 0:
                qtyA, qtyB = positions[pair]
                # Close existing long position at open
                transaction_costs_exit = calculate_transaction_costs(qtyA, qtyB, openA, openB, transaction_cost_pct)
                
                pnl_series.iloc[t] -= transaction_costs_exit
                pnl_series.iloc[t] += calculate_interday_long(
                    qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                )
                cash -= transaction_costs_exit
                cash += qtyA * openA  # sell A
                cash -= qtyB * openB  # buy B back.
                cash_series.iloc[t] = cash # TODO: REMOVE

                print(f'Exiting long spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, cash={cash}')
                positions[pair] = (0, 0)  # Reset positions
            # Case 7: -1 -> 0 : Exit short position
            elif prev_signal == -1 and signal == 0:
                qtyA, qtyB = positions[pair]
                # Close existing short position at open
                transaction_costs_exit = calculate_transaction_costs(qtyA, qtyB, openA, openB, transaction_cost_pct)
                
                pnl_series.iloc[t] -= transaction_costs_exit
                pnl_series.iloc[t] += calculate_interday_short(
                    qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                )
                cash -= transaction_costs_exit
                cash += qtyB * openB  # sell B
                cash -= qtyA * openA  # buy A back.
                cash_series.iloc[t] = cash # TODO: REMOVE
                print(f'Exiting short spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, cash={cash}')
                positions[pair] = (0, 0)  # Reset positions
            # Case 8: 1 -> -1 : Reverse from long to short
            elif prev_signal == 1 and signal == -1:
                # From long to short
                qtyA, qtyB = positions[pair]
                # 1. close existing long position at open ie 1 -> 0
                transaction_costs_exit = calculate_transaction_costs(qtyA, qtyB, openA, openB, transaction_cost_pct)
                
                pnl_series.iloc[t] -= transaction_costs_exit
                pnl_series.iloc[t] += calculate_interday_long(
                    qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                )
                cash -= transaction_costs_exit
                cash += qtyA * openA  # sell A
                cash -= qtyB * openB  # buy B back

                cash_series.iloc[t] = cash # TODO: REMOVE
                print(f'Flipping from long to short spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, cash={cash}')
                # 2. enter new short position at open ie 0 -> -1
                new_qtyA, new_qtyB = calculate_position_sizes(
                    cash, openA, openB, hedge_ratio, leverage=leverage
                )
                transaction_cost_entry = calculate_transaction_costs(new_qtyA, new_qtyB, openA, openB, transaction_cost_pct)
                purchase_cost = (new_qtyB * openB) - (new_qtyA * openA) # cost = buy B - sell A. if positive, cash outflow
                cash -= transaction_cost_entry
                cash -= purchase_cost
                pnl_series.iloc[t] -= transaction_cost_entry
                pnl_series.iloc[t] += calculate_intraday_short(
                    new_qtyA, new_qtyB, openA, openB, closeA, closeB
                )
                cash_series.iloc[t] = cash # TODO: REMOVE
                positions[pair] = (new_qtyA, new_qtyB)

                print(f'Flipped to short spread on {pair} at time {signals.index[t]}: qtyA={new_qtyA}, qtyB={new_qtyB}, openA={openA}, openB={openB}, position_cash = {(new_qtyB * openB) - (new_qtyA * openA)}, cash={cash}, hedge_ratio={hedge_ratio}')
            # Case 9: -1 -> 1 : Reverse from short to long
            elif prev_signal == -1 and signal == 1:
                # From short to long
                qtyA, qtyB = positions[pair]
                # 1. close existing short position at open
                transaction_costs_exit = calculate_transaction_costs(qtyA, qtyB, openA, openB, transaction_cost_pct)
                
                pnl_series.iloc[t] -= transaction_costs_exit
                pnl_series.iloc[t] += calculate_interday_short(
                    qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                )
                cash -= transaction_costs_exit
                cash += qtyB * openB  # sell B
                cash -= qtyA * openA  # buy A back.
                cash_series.iloc[t] = cash # TODO: REMOVE
                print(f'Flipping from short to long spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, cash={cash}')

                # 2. enter new long position at open
                new_qtyA, new_qtyB = calculate_position_sizes(
                    cash, openA, openB, hedge_ratio, leverage=leverage
                )
                transaction_cost_entry = calculate_transaction_costs(new_qtyA, new_qtyB, openA, openB, transaction_cost_pct)
                purchase_cost = (new_qtyA * openA) - (new_qtyB * openB) # cost = buy A - sell B. if positive, cash outflow
                cash -= transaction_cost_entry
                cash -= purchase_cost
                pnl_series.iloc[t] -= transaction_cost_entry
                pnl_series.iloc[t] += calculate_intraday_long(
                    new_qtyA, new_qtyB, openA, openB, closeA, closeB
                )
                cash_series.iloc[t] = cash # TODO: REMOVE
                positions[pair] = (new_qtyA, new_qtyB)
                print(f'Flipped to long spread on {pair} at time {signals.index[t]}: qtyA={new_qtyA}, qtyB={new_qtyB}, openA={openA}, openB={openB}, position_cash = {(new_qtyA * openA) - (new_qtyB * openB)}, cash={cash}, hedge_ratio={hedge_ratio}, leverage={leverage}')
        # end of all pairs for current time t
        cash_series.iloc[t] = cash


    # end of all time points
    # close out all open positions
    for pair, (qtyA, qtyB) in positions.items():
        if qtyA == 0 and qtyB == 0:
            continue  # No open position to close
        tickerA, tickerB = pair.split("_")
        closeA = close_data[tickerA].iloc[-1]
        closeB = close_data[tickerB].iloc[-1]

        if signals[pair].iloc[-1] == 1:
            # Closing long position
            transaction_costs_exit = calculate_transaction_costs(qtyA, qtyB, closeA, closeB, transaction_cost_pct)
            pnl_series.iloc[-1] -= transaction_costs_exit

            cash -= transaction_costs_exit
            cash += qtyA * closeA  # sell A
            cash -= qtyB * closeB  # buy B back.
            cash_series.iloc[-1] = cash
            print(f'Closing final long spread on {pair} at end: qtyA={qtyA}, qtyB={qtyB}, openA={closeA}, openB={closeB}, cash={cash}')
        elif signals[pair].iloc[-1] == -1:
            # Closing short position
            transaction_costs_exit = calculate_transaction_costs(qtyA, qtyB, closeA, closeB, transaction_cost_pct)
            pnl_series.iloc[-1] -= transaction_costs_exit
            
            cash -= transaction_costs_exit
            cash += qtyB * closeB  # sell B
            cash -= qtyA * closeA  # buy A back.
            cash_series.iloc[-1] = cash
            print(f'Closing final short spread on {pair} at end: qtyA={qtyA}, qtyB={qtyB}, openA={closeA}, openB={closeB}, cash={cash}')

           
    return pnl_series, cash_series



