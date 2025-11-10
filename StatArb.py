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

    hedge_ratio = hedge_ratio.apply(lambda x: abs(x))

    return signals, hedge_ratios


# --------------------------------------------
# CALCULATE PNL [CELL]
# --------------------------------------------


# ------------------------ HELPERS ------------------------
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
    pnlB = abs(qtyB) * (openB - closeB)
    return pnlA + pnlB


def calculate_intraday_short(
    qtyA: float, qtyB: float, openA: float, openB: float, closeA: float, closeB: float
) -> float:
    """
    Calculates intraday PnL when going short on spread. This represents a short pos in A and a long pos in B.

    Given as open - close for A and close - open for B.
    """
    pnlA = abs(qtyA) * (openA - closeA)
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
    pnlB = abs(qtyB) * (closeB_prev - openB)
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
    pnlA = abs(qtyA) * (closeA_prev - openA)
    pnlB = qtyB * (openB - closeB_prev)
    return pnlA + pnlB


# ------------------------ HELPERS END ------------------------
def calculate_pnl_from_signals_and_hedge_ratios(
    signals: pd.DataFrame,
    hedge_ratios: pd.DataFrame,
    stock_open_data: pd.DataFrame,
    stock_close_data: pd.DataFrame,
    cash: float = 500_000,
    transaction_cost_pct: float = 0.0,
) -> pd.Series:
    """
    Iterates through signals and hedge ratios to calculate PnL. Each row represents 1 time period
    -> sanity check: len(signals) == len(hedge_ratios) == len(stock_data)

    inner loop iterates through each pair of stocks.
        Check signals for pair p at time t:

    Returns: pd.Series of PnL over time.
    """
    # sanity check
    assert (
        len(signals)
        == len(hedge_ratios)
        == len(stock_close_data)
        == len(stock_open_data)
    ), "DataFrames must have the same length"

    # init data structures
    positions = create_positions_dict(tickers)
    pnl_series = pd.Series(0.0, index=signals.index)
    cash_series = pd.Series(0.0, index=signals.index)

    # iterate through time
    # starting at t=2. T will be the first signal, but we only execute at T+1
    for t in range(len(signals)):
        current_signals = signals.iloc[t]
        prev_signals = (
            signals.iloc[t - 1] if t > 0 else pd.Series(0, index=signals.columns)
        )

        current_hedge_ratios = hedge_ratios.iloc[t]

        current_close = stock_close_data.iloc[t]
        # error handling, but we should not be looking up this series
        prev_close = (
            stock_close_data.iloc[t - 1]
            if t > 0
            else pd.Series(0, index=stock_close_data.columns)
        )

        current_open = stock_open_data.iloc[t]

        # error handling: skip if current or prev signals are NaN
        for pair in signals.columns:
            ticker_a, ticker_b = pair.split("_")
            signal = current_signals[pair]
            prev_signal = prev_signals[pair]
            hedge_ratio = current_hedge_ratios[pair]

            openA = current_open[ticker_a]
            openB = current_open[ticker_b]
            closeA = current_close[ticker_a]
            closeB = current_close[ticker_b]
            closeA_prev = prev_close[ticker_a]
            closeB_prev = prev_close[ticker_b]

            # check signals
            # TODO: Handle leverage effect on cash. Currently affects position size, but deducts the entire position cost from cash. Need to only deduct initial margin.
            # Case 1 & 2: 0 -> something
            if prev_signal == 0 and signal != 0:
                # calculate position sizes & transaction costs, regardless of long/short
                qtyA, qtyB = calculate_position_sizes(
                    cash, openA, openB, hedge_ratio, leverage=leverage
                )
                transaction_costs = calculate_transaction_costs(
                    qtyA, qtyB, openA, openB, transaction_cost_pct
                )

                # sanity check: ensure we have enough cash
                if transaction_costs > cash:
                    continue  # skip this trade if not enough cash

                if signal == 1:
                    # Long the spread
                    # Long A, Short B
                    # pnl -> intraday long since we entered at today's open
                    positions[pair] = (qtyA, -qtyB)
                    cash -= transaction_costs
                    # cash -> long costs, short proceeds
                    purchase_cost = (qtyA * openA) - (
                        qtyB * openB
                    )  # cost = buy A - sell B
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] -= transaction_costs
                    pnl_series.iloc[t] += calculate_intraday_long(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )
                    print(
                        f"Entered long spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, position_cash = {(qtyA * openA) - (qtyB * openB)}, cash={cash}, hedge_ratio={hedge_ratio}, leverage={leverage}"
                    )
                    print(
                        f"Intraday long PnL: {calculate_intraday_long(qtyA, qtyB, openA, openB, closeA, closeB)}"
                    )

                elif signal == -1:
                    # Short the spread
                    # Short A, Long B
                    # pnl -> intraday short since we entered at today's open
                    positions[pair] = (-qtyA, qtyB)
                    cash -= transaction_costs
                    # cash -> short proceeds, long costs
                    purchase_cost = (qtyB * openB) - (
                        qtyA * openA
                    )  # cost = buy B - sell A
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] -= transaction_costs
                    pnl_series.iloc[t] += calculate_intraday_short(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )

                    print(
                        f"Entered short spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, position_cash = {(qtyB * openB) - (qtyA * openA)}, cash={cash}, hedge_ratio={hedge_ratio}"
                    )
                    print(
                        f"Intraday short PnL: {calculate_intraday_short(qtyA, qtyB, openA, openB, closeA, closeB)}"
                    )

            # Case 3 & 4: something -> 0
            elif prev_signal != 0 and signal == 0:
                qtyA, qtyB = positions[pair]
                transaction_costs = calculate_transaction_costs(
                    qtyA, qtyB, openA, openB, transaction_cost_pct
                )

                if prev_signal == 1:
                    # Exiting long spread at Open today
                    # calculate pre-market price movement PnL
                    pnl_series.iloc[t] += calculate_interday_long(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    pnl_series.iloc[t] -= transaction_costs
                    # recognise cash from closing positions at open
                    cash -= transaction_costs
                    cash += qtyA * openA  # sell A
                    cash += (
                        qtyB * openB
                    )  # buy B back. Add to cash because qtyB is negative, so this is effectively a subtraction
                    # reset positions
                    positions[pair] = (0, 0)

                elif prev_signal == -1:
                    # Exiting short spread at Open today
                    # calculate pre-market price movement PnL
                    pnl_series.iloc[t] += calculate_interday_short(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    pnl_series.iloc[t] -= transaction_costs
                    # recognise cash from closing positions at open
                    cash -= transaction_costs
                    cash += qtyB * openB  # sell B
                    cash += (
                        qtyA * openA
                    )  # buy A back. Add to cash because qtyA is negative, so this is effectively a subtraction
                    # reset positions
                    positions[pair] = (0, 0)

            # case 5 and 6: 1 -> 1 or -1 -> -1
            elif prev_signal == signal and signal != 0:
                # maintain position, but calculate day to day PnL
                qtyA, qtyB = positions[pair]
                if signal == 1:
                    # Currently long spread, calculate interday PnL & intraday PnL
                    price_movement = calculate_interday_long(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    ) + calculate_intraday_long(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )
                    pnl_series.iloc[t] += price_movement
                elif signal == -1:
                    # currently short spread
                    price_movement = calculate_interday_short(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    ) + calculate_intraday_short(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )
                    pnl_series.iloc[t] += price_movement

            # case 7 and 8: 1 -> -1 or -1 -> 1
            # case 9 (0 -> 0) implicitly handled by doing nothing
            else:

                # Case 7
                if prev_signal == 1 and signal == -1:
                    # From long to short
                    qtyA, qtyB = positions[pair]
                    # 1. close existing long position at open
                    transaction_cost = calculate_transaction_costs(
                        qtyA, qtyB, openA, openB, transaction_cost_pct
                    )

                    pnl_series.iloc[t] -= transaction_cost
                    pnl_series.iloc[t] += calculate_interday_long(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    cash -= transaction_cost
                    cash += qtyA * openA  # sell A
                    cash += (
                        qtyB * openB
                    )  # buy B back. Add to cash because qtyB is negative, so this is effectively a subtraction

                    # 2. Enter short position
                    new_qtyA, new_qtyB = calculate_position_sizes(
                        cash, openA, openB, hedge_ratio, leverage=leverage
                    )
                    transaction_cost = calculate_transaction_costs(
                        new_qtyA, new_qtyB, openA, openB, transaction_cost_pct
                    )

                    # sanity check : ensure we have enough cash
                    if transaction_cost > cash:
                        continue  # skip this trade if not enough cash

                    purchase_cost = (new_qtyB * openB) - (
                        new_qtyA * openA
                    )  # cost = buy B - sell A. if positive, cash outflow
                    cash -= transaction_cost
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= transaction_cost
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] += calculate_intraday_short(
                        new_qtyA, new_qtyB, openA, openB, closeA, closeB
                    )

                    positions[pair] = (-new_qtyA, new_qtyB)

                # Case 8
                elif prev_signal == -1 and signal == 1:
                    # From short to long
                    qtyA, qtyB = positions[pair]
                    # 1. close existing short position at open
                    transaction_cost = calculate_transaction_costs(
                        qtyA, qtyB, openA, openB, transaction_cost_pct
                    )

                    pnl_series.iloc[t] -= transaction_cost
                    pnl_series.iloc[t] += calculate_interday_short(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    cash -= transaction_cost
                    cash += qtyB * openB  # sell B
                    cash += (
                        qtyA * openA
                    )  # buy A back. Add to cash because qtyA is negative, so this is effectively a subtraction

                    # 2. Enter long position
                    new_qtyA, new_qtyB = calculate_position_sizes(
                        cash, openA, openB, hedge_ratio, leverage=leverage
                    )
                    transaction_cost = calculate_transaction_costs(
                        new_qtyA, new_qtyB, openA, openB, transaction_cost_pct
                    )

                    # sanity check: ensure we have enough cash
                    if transaction_cost > cash:
                        continue  # skip this trade if not enough cash

                    purchase_cost = (new_qtyA * openA) - (
                        new_qtyB * openB
                    )  # cost = buy A - sell B. if positive, cash outflow
                    cash -= transaction_cost
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= transaction_cost
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] += calculate_intraday_long(
                        new_qtyA, new_qtyB, openA, openB, closeA, closeB
                    )

                    positions[pair] = (new_qtyA, -new_qtyB)
        cash_series.iloc[t] = cash

    # final sanity check: ensure all positions are closed at end
    for pair, (qtyA, qtyB) in positions.items():
        if qtyA == qtyB == 0:
            continue
        else:
            qtyA, qtyB = positions[pair]
            closeA = stock_close_data.iloc[-1][pair.split("_")[0]]
            closeB = stock_close_data.iloc[-1][pair.split("_")[1]]
            transaction_costs = calculate_transaction_costs(
                qtyA, qtyB, closeA, closeB, transaction_cost_pct
            )
            pnl_series.iloc[
                -1
            ] -= transaction_costs  # dont update intraday pnl since it was already calculated
            cash -= transaction_costs
            if qtyA > 0:
                # closing long spread
                cash += qtyA * closeA  # sell A
                cash += (
                    qtyB * closeB
                )  # buy B back. Add to cash because qtyB is negative, so this is effectively a subtraction
            else:
                # closing short spread
                cash += qtyB * closeB  # sell B
                cash += (
                    qtyA * closeA
                )  # buy A back. Add to cash because qtyA is negative, so this is effectively a subtraction
            positions[pair] = (0, 0)
    cash_series.iloc[-1] = cash

    return pnl_series, cash_series


# ----------------------------------
# Performance Metrics and Plots
# ----------------------------------


def get_performance_metrics(
    pnl_series: pd.Series, benchmark_ticker="SPY", cash=500_000
) -> dict:
    """
    Calculates performance metrics from PnL series.

    Returns a dictionary of performance metrics.
    """

    # 1. Calculate Portfolio Value Time Series
    # Portfolio value = Initial cash + cumulative PnL
    portfolio_value = cash + pnl_series.cumsum()

    # 2. Total Return ($) and Total Return (%)
    total_return = portfolio_value.iloc[-1] - cash  # Total PnL ($)
    total_return_pct = (total_return / cash) * 100  # Total return (%)

    # 3. Annualized Return (%)
    num_days = len(pnl_series)
    annualized_return = (
        (portfolio_value.iloc[-1] / cash) ** (252 / num_days)
    ) - 1  # Annualized Return Formula

    # 4. Daily Returns from Portfolio Value
    daily_returns = portfolio_value.pct_change().dropna()  # Daily returns in percentage

    # 5. Annualized Volatility (Standard Deviation %)
    annualized_volatility = daily_returns.std() * np.sqrt(
        252
    )  # Annualized standard deviation

    # 6. Sharpe Ratio
    if annualized_volatility != 0:
        sharpe_ratio = annualized_return / annualized_volatility
    else:
        sharpe_ratio = np.nan

    # 7. Sortino Ratio (using downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_volatility = downside_returns.std() * np.sqrt(
        252
    )  # Annualized downside deviation
    if downside_volatility != 0:
        sortino_ratio = annualized_return / downside_volatility
    else:
        sortino_ratio = np.nan

    # 8. Calmar Ratio (Annualized Return / Max Drawdown)
    max_drawdown = calculate_max_drawdown(portfolio_value)
    calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan

    # 9. Treynor Ratio (Annualized Return / Beta)
    beta = calculate_beta(pnl_series, benchmark_ticker)
    treynor_ratio = annualized_return / beta if beta != 0 else np.nan

    # 10. Information Ratio (Annualized Excess Return over Benchmark / Tracking Error)
    benchmark_data = yf.download(
        benchmark_ticker, start=pnl_series.index[0], end=pnl_series.index[-1]
    )["Close"]
    benchmark_returns = benchmark_data.pct_change().dropna()  # Benchmark returns
    excess_returns = daily_returns - benchmark_returns  # Excess returns
    tracking_error = excess_returns.std() * np.sqrt(252)  # Tracking error
    info_ratio = annualized_return / tracking_error if tracking_error != 0 else np.nan

    # 11. Max Drawdown (%)
    max_drawdown_pct = max_drawdown * 100  # Percentage

    # 12. Max Drawdown Duration (days)
    max_drawdown_duration = calculate_max_drawdown_duration(portfolio_value)

    # 13. Beta
    beta = calculate_beta(pnl_series, benchmark_ticker)

    # 14. Alpha (assuming risk-free rate is 0)
    alpha = calculate_alpha(pnl_series, benchmark_ticker)

    return {
        "Total Return ($)": total_return,
        "Total Return (%)": total_return_pct,
        "Annualized Return (%)": annualized_return * 100,
        "Annualized Volatility (%)": annualized_volatility * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Calmar Ratio": calmar_ratio,
        "Treynor Ratio": treynor_ratio,
        "Information Ratio": info_ratio,
        "Max Drawdown (%)": max_drawdown_pct,
        "Max Drawdown Duration (days)": max_drawdown_duration,
        "Beta": beta,
        "Alpha": alpha,
    }


# Helper functions for max drawdown, beta, and alpha calculations:


def calculate_max_drawdown(portfolio_value: pd.Series) -> float:
    """Calculate the maximum drawdown as a percentage of total capital."""
    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown


def calculate_max_drawdown_duration(portfolio_value: pd.Series) -> int:
    """Calculate the duration of the maximum drawdown."""
    peak = portfolio_value.cummax()
    drawdown = (portfolio_value - peak) / peak
    drawdown_duration = (drawdown < 0).sum()
    return drawdown_duration


def calculate_beta(pnl_series: pd.Series, benchmark_ticker: str) -> float:
    """Calculate the beta (sensitivity to benchmark) using covariance and variance."""
    # Download benchmark data (e.g., SPY) for the same period as the pnl_series
    benchmark_data = yf.download(
        benchmark_ticker, start=pnl_series.index[0], end=pnl_series.index[-1]
    )["Close"]

    # Calculate daily returns for both the stock (PnL series) and the benchmark
    stock_returns = (
        pnl_series / pnl_series.shift(1) - 1
    )  # Calculate daily returns from PnL series
    benchmark_returns = (
        benchmark_data.pct_change().dropna()
    )  # Calculate benchmark daily returns

    # Ensure both series are aligned (same index)
    aligned_data = stock_returns.align(benchmark_returns, join="inner")
    stock_returns = aligned_data[0]
    benchmark_returns = aligned_data[1]

    # Calculate covariance between stock returns and benchmark returns
    covariance = np.cov(stock_returns, benchmark_returns)[0, 1]  # Covariance

    # Calculate variance of benchmark returns
    benchmark_variance = benchmark_returns.var()  # Variance of the benchmark returns

    # Calculate beta (covariance / variance)
    if benchmark_variance != 0:
        beta = covariance / benchmark_variance
    else:
        beta = np.nan  # If benchmark variance is zero, beta can't be calculated

    return beta


def calculate_alpha(pnl_series: pd.Series, benchmark_ticker: str) -> float:
    """Calculate the alpha (excess return) of the portfolio."""
    beta = calculate_beta(pnl_series, benchmark_ticker)
    benchmark_data = yf.download(
        benchmark_ticker, start=pnl_series.index[0], end=pnl_series.index[-1]
    )["Close"]
    benchmark_returns = benchmark_data.pct_change().dropna()
    excess_returns = pnl_series / pnl_series.shift(1) - 1 - beta * benchmark_returns
    alpha = excess_returns.mean() * 252  # Annualize alpha
    return alpha


# ------------------------
# Plot Figures
# ------------------------


def plot_figures(
    pnl_series: pd.Series,
    benchmark_ticker: str,
    tickers: List[str],
    cash: float = 500_000,
) -> None:
    """
    Plots various performance figures:
    1. Histogram of strategy returns (daily pct change).
    2. Strategy vs. Benchmark performance (SPY) and Long on Basket.
    3. Max Drawdown vs. Portfolio Cumulative Log Return.
    4. Equity Curve (Portfolio Value over time).
    """

    # Ensure the pnl_series is properly aligned and the index is datetime
    pnl_series = pnl_series.sort_index()

    # 1. Histogram of Strategy Returns
    daily_returns = pnl_series / cash  # Convert PnL to daily returns in % terms
    plt.figure(figsize=(10, 6))
    plt.hist(daily_returns, bins=30, alpha=0.75, color="blue", label="Strategy Returns")
    plt.title("Histogram of Strategy Returns")
    plt.xlabel("Daily Returns")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.legend(loc="best")
    plt.show()

    # 2. Strategy vs. Benchmark (SPY) vs. Long on Basket
    # Download SPY data and calculate returns
    benchmark_data = yf.download(
        benchmark_ticker, start=pnl_series.index[0], end=pnl_series.index[-1]
    )["Close"]
    benchmark_returns = benchmark_data.pct_change().dropna()

    # Calculate returns for the basket (long all stocks)
    basket_data = yf.download(
        tickers, start=pnl_series.index[0], end=pnl_series.index[-1]
    )["Close"]
    basket_returns = (
        basket_data.pct_change().mean(axis=1).dropna()
    )  # Simple mean of all stock returns

    # Calculate cumulative returns for the strategy, SPY, and long basket
    strategy_cumulative_returns = (1 + daily_returns).cumprod() - 1
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1
    basket_cumulative_returns = (1 + basket_returns).cumprod() - 1

    # Plot strategy vs benchmark vs long basket
    plt.figure(figsize=(10, 6))
    plt.plot(strategy_cumulative_returns, label="Strategy", color="blue")
    plt.plot(
        benchmark_cumulative_returns,
        label=f"{benchmark_ticker} (Benchmark)",
        color="red",
        linestyle="--",
    )
    plt.plot(
        basket_cumulative_returns, label="Long Basket", color="green", linestyle=":"
    )
    plt.title("Strategy vs Benchmark vs Long Basket")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Max Drawdown vs Portfolio Cumulative Log Return
    # Calculate cumulative log returns of the strategy
    log_returns = np.log(1 + daily_returns)
    cumulative_log_returns = log_returns.cumsum()

    # Calculate max drawdown
    peak = cumulative_log_returns.cummax()
    drawdown = cumulative_log_returns - peak
    max_drawdown = drawdown.min()

    # Plot max drawdown vs cumulative log return
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_log_returns, label="Cumulative Log Return", color="blue")
    plt.plot(
        drawdown, label=f"Max Drawdown: {max_drawdown:.2%}", color="red", linestyle="--"
    )
    plt.title("Max Drawdown vs Portfolio Cumulative Log Return")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. Equity Curve (Portfolio Value over Time)
    equity_curve = (1 + daily_returns).cumprod() * cash  # Portfolio value over time
    plt.figure(figsize=(10, 6))
    plt.plot(equity_curve, label="Equity Curve (Portfolio Value)", color="green")
    plt.title("Equity Curve (Portfolio Value Over Time)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


### FOR NATHAN
# changes logic to store positive quantities for positions at all times


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

    print(
        f"[DEBUG POS SIZE] - Starting cash: {total_cash}, position_size: {position_size}"
    )
    print(f"[DEBUG POS SIZE] - Calculated qtyA: {qtyA}, qtyB: {qtyB}")
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


# ------------------------ HELPERS END ------------------------
def calculate_pnl_from_signals_and_hedge_ratios(
    signals: pd.DataFrame,
    tickers: pd.Series,
    hedge_ratios: pd.DataFrame,
    stock_open_data: pd.DataFrame,
    stock_close_data: pd.DataFrame,
    cash: float = 500_000,
    transaction_cost_pct: float = 0.0,
    leverage: float = 1.0,
) -> pd.Series:
    """
    Iterates through signals and hedge ratios to calculate PnL. Each row represents 1 time period
    -> sanity check: len(signals) == len(hedge_ratios) == len(stock_data)

    inner loop iterates through each pair of stocks.
        Check signals for pair p at time t:

    Returns: pd.Series of PnL over time.
    """
    # sanity check
    assert (
        len(signals)
        == len(hedge_ratios)
        == len(stock_close_data)
        == len(stock_open_data)
    ), "DataFrames must have the same length"

    # init data structures
    positions = create_positions_dict(tickers)
    pnl_series = pd.Series(0.0, index=signals.index)
    cash_series = pd.Series(0.0, index=signals.index)

    # iterate through time
    # starting at t=2. T will be the first signal, but we only execute at T+1
    for t in range(len(signals)):
        current_signals = signals.iloc[t]
        prev_signals = (
            signals.iloc[t - 1] if t > 0 else pd.Series(0, index=signals.columns)
        )

        current_hedge_ratios = hedge_ratios.iloc[t]

        current_close = stock_close_data.iloc[t]
        # error handling, but we should not be looking up this series
        prev_close = (
            stock_close_data.iloc[t - 1]
            if t > 0
            else pd.Series(0, index=stock_close_data.columns)
        )

        current_open = stock_open_data.iloc[t]

        # error handling: skip if current or prev signals are NaN
        for pair in signals.columns:
            ticker_a, ticker_b = pair.split("_")
            signal = current_signals[pair]
            prev_signal = prev_signals[pair]
            hedge_ratio = current_hedge_ratios[pair]

            openA = current_open[ticker_a]
            openB = current_open[ticker_b]
            closeA = current_close[ticker_a]
            closeB = current_close[ticker_b]
            closeA_prev = prev_close[ticker_a]
            closeB_prev = prev_close[ticker_b]

            # check signals
            # TODO: Handle leverage effect on cash. Currently affects position size, but deducts the entire position cost from cash. Need to only deduct initial margin.
            # Case 1 & 2: 0 -> something
            if prev_signal == 0 and signal != 0:
                # calculate position sizes & transaction costs, regardless of long/short
                print(f"Opening cash: {cash}")
                qtyA, qtyB = calculate_position_sizes(
                    cash, openA, openB, hedge_ratio, leverage
                )
                transaction_costs = calculate_transaction_costs(
                    qtyA, qtyB, openA, openB, transaction_cost_pct
                )

                # sanity check: ensure we have enough cash
                if transaction_costs > cash:
                    continue  # skip this trade if not enough cash

                if signal == 1:
                    # Long the spread
                    # Long A, Short B
                    # pnl -> intraday long since we entered at today's open
                    positions[pair] = (qtyA, qtyB)
                    cash -= transaction_costs
                    # cash -> long costs, short proceeds
                    purchase_cost = (qtyA * openA) - (
                        qtyB * openB
                    )  # cost = buy A - sell B
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] -= transaction_costs
                    pnl_series.iloc[t] += calculate_intraday_long(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )
                    cash_series.iloc[t] = cash  # TODO: REMOVE
                    print(
                        f"Entered long spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, position_cash = {(qtyA * openA) - (qtyB * openB)}, cash={cash}, hedge_ratio={hedge_ratio}"
                    )
                    print(
                        f"Intraday long PnL: {calculate_intraday_long(qtyA, qtyB, openA, openB, closeA, closeB)}"
                    )

                elif signal == -1:
                    # Short the spread
                    # Short A, Long B
                    # pnl -> intraday short since we entered at today's open
                    positions[pair] = (qtyA, qtyB)
                    cash -= transaction_costs
                    # cash -> short proceeds, long costs
                    purchase_cost = (qtyB * openB) - (
                        qtyA * openA
                    )  # cost = buy B - sell A
                    cash -= purchase_cost
                    cash_series.iloc[t] = cash  # TODO: REMOVE
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] -= transaction_costs
                    pnl_series.iloc[t] += calculate_intraday_short(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )

                    print(
                        f"Entered short spread on {pair} at time {signals.index[t]}: qtyA={qtyA}, qtyB={qtyB}, openA={openA}, openB={openB}, position_cash = {(qtyB * openB) - (qtyA * openA)}, cash={cash}, hedge_ratio={hedge_ratio}"
                    )
                    print(
                        f"Intraday short PnL: {calculate_intraday_short(qtyA, qtyB, openA, openB, closeA, closeB)}"
                    )

            # Case 3 & 4: something -> 0
            elif prev_signal != 0 and signal == 0:
                qtyA, qtyB = positions[pair]
                transaction_costs = calculate_transaction_costs(
                    qtyA, qtyB, openA, openB, transaction_cost_pct
                )

                if prev_signal == 1:
                    # Exiting long spread at Open today
                    # calculate pre-market price movement PnL
                    pnl_series.iloc[t] += calculate_interday_long(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    pnl_series.iloc[t] -= transaction_costs
                    # recognise cash from closing positions at open
                    cash -= transaction_costs
                    cash += qtyA * openA  # sell A
                    cash -= qtyB * openB
                    cash_series.iloc[t] = cash  # TODO: REMOVE
                    # reset positions
                    positions[pair] = (0, 0)

                elif prev_signal == -1:
                    # Exiting short spread at Open today
                    # calculate pre-market price movement PnL
                    pnl_series.iloc[t] += calculate_interday_short(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    pnl_series.iloc[t] -= transaction_costs
                    # recognise cash from closing positions at open
                    cash -= transaction_costs
                    cash += qtyB * openB  # sell B
                    cash -= qtyA * openA
                    cash_series.iloc[t] = cash  # TODO: REMOVE
                    # reset positions
                    positions[pair] = (0, 0)

            # case 5 and 6: 1 -> 1 or -1 -> -1
            elif prev_signal == signal and signal != 0:
                # maintain position, but calculate day to day PnL
                qtyA, qtyB = positions[pair]
                if signal == 1:
                    # Currently long spread, calculate interday PnL & intraday PnL
                    price_movement = calculate_interday_long(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    ) + calculate_intraday_long(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )
                    pnl_series.iloc[t] += price_movement
                elif signal == -1:
                    # currently short spread
                    price_movement = calculate_interday_short(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    ) + calculate_intraday_short(
                        qtyA, qtyB, openA, openB, closeA, closeB
                    )
                    pnl_series.iloc[t] += price_movement
            # case 7 and 8: 1 -> -1 or -1 -> 1
            # case 9 (0 -> 0) implicitly handled by doing nothing
            else:
                # Case 7
                if prev_signal == 1 and signal == -1:
                    # From long to short
                    qtyA, qtyB = positions[pair]
                    # 1. close existing long position at open
                    transaction_costs_exit = calculate_transaction_costs(
                        qtyA, qtyB, openA, openB, transaction_cost_pct
                    )

                    pnl_series.iloc[t] -= transaction_costs_exit
                    pnl_series.iloc[t] += calculate_interday_long(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    cash -= transaction_costs_exit
                    cash += qtyA * openA  # sell A
                    cash += (
                        qtyB * openB
                    )  # buy B back. Add to cash because qtyB is negative, so this is effectively a subtraction
                    cash_series.iloc[t] = cash  # TODO: REMOVE

                    # 2. Enter short position
                    new_qtyA, new_qtyB = calculate_position_sizes(
                        cash, openA, openB, hedge_ratio, leverage
                    )
                    transaction_cost_entry = calculate_transaction_costs(
                        new_qtyA, new_qtyB, openA, openB, transaction_cost_pct
                    )
                    purchase_cost = (new_qtyB * openB) - (
                        new_qtyA * openA
                    )  # cost = buy B - sell A. if positive, cash outflow
                    cash -= transaction_cost_entry
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= transaction_cost_entry
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] += calculate_intraday_short(
                        new_qtyA, new_qtyB, openA, openB, closeA, closeB
                    )
                    cash_series.iloc[t] = cash  # TODO: REMOVE

                    positions[pair] = (new_qtyA, new_qtyB)

                # Case 8
                elif prev_signal == -1 and signal == 1:
                    # From short to long
                    qtyA, qtyB = positions[pair]
                    # 1. close existing short position at open
                    transaction_cost_exit = calculate_transaction_costs(
                        qtyA, qtyB, openA, openB, transaction_cost_pct
                    )

                    pnl_series.iloc[t] -= transaction_cost_exit
                    pnl_series.iloc[t] += calculate_interday_short(
                        qtyA, qtyB, closeA_prev, closeB_prev, openA, openB
                    )
                    cash -= transaction_cost_exit
                    cash += qtyB * openB  # sell B
                    cash += (
                        qtyA * openA
                    )  # buy A back. Add to cash because qtyA is negative, so this is effectively a subtraction

                    # 2. Enter long position
                    new_qtyA, new_qtyB = calculate_position_sizes(
                        cash, openA, openB, hedge_ratio, leverage
                    )
                    transaction_cost_entry = calculate_transaction_costs(
                        new_qtyA, new_qtyB, openA, openB, transaction_cost_pct
                    )
                    purchase_cost = (new_qtyA * openA) - (
                        new_qtyB * openB
                    )  # cost = buy A - sell B. if positive, cash outflow
                    cash -= transaction_cost_entry
                    cash -= purchase_cost
                    pnl_series.iloc[t] -= transaction_cost_entry
                    pnl_series.iloc[t] -= purchase_cost
                    pnl_series.iloc[t] += calculate_intraday_long(
                        new_qtyA, new_qtyB, openA, openB, closeA, closeB
                    )
                    cash_series.iloc[t] = cash  # TODO: REMOVE
                    positions[pair] = (new_qtyA, -new_qtyB)

    return pnl_series, cash_series
