import pandas as pd
import numpy as np
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler 
from keras.models import load_model # type: ignore

def get_sp500_tickers():
    """
    Gets a list of S&P500 ticker symbols from Wikipedia for the user to select
    
    Returns:
    - List of S&P500 ticker symbols
    """

    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].tolist()
    return tickers

def df_for_candlestick(horizon, ticker):
    """
    Creates dataframe to plot candlestick chart.
    
    Parameters:
    - horizon: Horizon for the candlestick chart (e.g., '1d', '5d', '1mo', '6mo', '1y', '5y')
    - ticker: Selected ticker symbol

    Returns:
    - Candlestick chart dataframe
    """

    stock = yf.Ticker(ticker)
    candle_dataframe = stock.history(period=f'{horizon}')
    return candle_dataframe

def plot_candlestick(ticker, horizon, candle_dataframe):
    """
    Creates the candlestick chart plot for a given ticker symbol.
    
    Parameters:
    - ticker: Selected ticker symbol
    - horizon: Horizon for the candlestick chart (e.g., '1d', '5d', '1mo', '6mo', '1y', '5y')
    - candle_dataframe: Candlestick chart dataframe

    Returns:
    - Candlestick chart dataframe
    """

    fig = go.Figure(data=[go.Candlestick(x=candle_dataframe.index,
                open=candle_dataframe['Open'],
                high=candle_dataframe['High'],
                low=candle_dataframe['Low'],
                close=candle_dataframe['Close'])])

    fig.update_layout(title=f'{ticker} {horizon} Chart')
    return fig

def create_stats_dataframe(ticker, keys_to_extract, stat):
    """
    Creates stock statistics dataframe
    
    Parameters:
    - selected_ticker: Selected ticker symbol
    - keys_to_extract: List of keys to extract from Yahoo Finance Ticker get_info() function
    - stat: Type of statistic (Stock, Income Statement, Balance Sheet, Cash Flow, Share, Dividends and Splits)

    Returns:
    - Statistics dataframe
    """

    ticker = yf.Ticker(ticker)
    dataframe = ticker.get_info()
    dict = {key: dataframe[key] for key in keys_to_extract if key in dataframe}
    df = pd.DataFrame(dict.items(), columns=['Key', 'Value'])
    df.columns = [f'{stat} Statistics', 'Value'] 
    df.index = range(1, len(df) + 1)

    return df

def income_statement(ticker, timeframe):
    """
    Creates income statement dataframe
    
    Parameters:
    - ticker: Selected ticker symbol
    - timeframe: Type of income statement (yearly or quarterly)

    Returns:
    - Income statement dataframe
    """

    ticker = yf.Ticker(ticker)
    dataframe = ticker.get_income_stmt(freq=timeframe)
    dataframe.columns = dataframe.columns.strftime('%Y-%m-%d')
    if timeframe == 'yearly':
        income_statement = dataframe.drop(columns=dataframe.columns[-1], axis=1)
        return income_statement
    elif timeframe == 'quarterly':
        income_statement = dataframe.drop(columns=dataframe.columns[-2:], axis=1)
        return income_statement

def balance_sheet(ticker, timeframe):
    """
    Creates balance sheet dataframe
    
    Parameters:
    - ticker: Selected ticker symbol
    - timeframe: Type of balance sheet (yearly or quarterly)

    Returns:
    - Balance sheet dataframe
    """

    ticker = yf.Ticker(ticker)
    dataframe = ticker.get_balance_sheet(freq=timeframe)
    dataframe.columns = dataframe.columns.strftime('%Y-%m-%d')
    if timeframe == 'yearly':
        balance_sheet = dataframe.drop(columns=dataframe.columns[-1], axis=1)
        return balance_sheet
    elif timeframe == 'quarterly':
        balance_sheet = dataframe.drop(columns=dataframe.columns[-2:], axis=1)
        return balance_sheet
    
def calculate_current_ratio(dataframe):
    """
    Calculates current ratio using balance sheet dataframe
    
    Parameters:
    - dataframe: Balance sheet dataframe

    Returns:
    - Current ratio value
    """

    filtered_df = dataframe.loc[['CurrentAssets', 'CurrentLiabilities']]
    current_assets = filtered_df.iloc[:,0].values[0]
    current_liabilities = filtered_df.iloc[:,0].values[1]
    current_ratio = current_assets / current_liabilities 
    current_ratio = np.round(current_ratio, 2)

    return current_ratio

def cashflow(ticker, timeframe):
    """
    Creates cashflow dataframe
    
    Parameters:
    - ticker: Selected ticker symbol
    - timeframe: Type of cashflow (yearly or quarterly)

    Returns:
    - Cashflow dataframe
    """

    ticker = yf.Ticker(ticker)
    dataframe = ticker.get_cashflow(freq=timeframe)
    dataframe.columns = dataframe.columns.strftime('%Y-%m-%d')
    if timeframe == 'yearly':
        cashflow = dataframe.drop(columns=dataframe.columns[-1], axis=1)
        return cashflow
    elif timeframe == 'quarterly':
        cashflow = dataframe.drop(columns=dataframe.columns[-2:], axis=1)
        return cashflow
    
def earnings_history(ticker):
    """
    Creates earnings history dataframe
    
    Parameters:
    - ticker: Selected ticker symbol

    Returns:
    - Earnings history dataframe
    """

    ticker = yf.Ticker(ticker)
    dataframe = ticker.get_earnings_history()
    dataframe['surprisePercent'] = dataframe['surprisePercent'].values * 100
    dataframe.index = dataframe.index.strftime('%Y-%m-%d')
    return dataframe

def plot_income_statement(dataframe):
    """
    Creates bar plot of the yearly or quarterly net income, gross profit and total revenue.
    
    Parameters:
    - dataframe: Income statement dataframe

    Returns:
    - Plotly bar plot displaying net income, gross profit and total revenue on y-axis and year or quarter on the x-axis
    """

    filtered_df = dataframe.loc[['NetIncome', 'GrossProfit', 'TotalRevenue']]
    filtered_df = filtered_df.T.reset_index()
    filtered_df.rename(columns={"index": "Date"}, inplace=True)
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"]).dt.date
    filtered_df = filtered_df.loc[::-1]
    df_melted = filtered_df.melt(id_vars="Date", var_name="Metric", value_name="Amount")

    fig = px.bar(
    df_melted,
    title='Net Income, Gross Profit and Total Revenue',
    x="Date",
    y="Amount",
    color="Metric",
    barmode="group",
    labels={"Amount": "Amount ($)", "Date": "Date", "Metric": "Financial Metric"}
    )

    fig.update_layout(
        xaxis=dict(type="category"),
        yaxis=dict(tickformat=","),
        bargap=0.2
    )

    return fig

def plot_assets_liabilities(dataframe):
    """
    Creates bar plot of yearly or quarterly assets and liabilities
    
    Parameters:
    - dataframe: Balance sheet dataframe

    Returns:
    - Plotly bar plot of yearly or quarterly assets and liabilities 
    """

    filtered_df = dataframe.loc[['TotalAssets', 'TotalLiabilitiesNetMinorityInterest']]
    filtered_df = filtered_df.T.reset_index()
    filtered_df.rename(columns={"index": "Date"}, inplace=True)
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"]).dt.date
    filtered_df = filtered_df.loc[::-1]
    filtered_df.rename(columns={"TotalLiabilitiesNetMinorityInterest": "TotalLiabilities"}, inplace=True)

    df_melted = filtered_df.melt(id_vars="Date", var_name="Metric", value_name="Amount")

    fig = px.bar(
        df_melted,
        x="Date",
        y="Amount",
        color="Metric",
        barmode="group",
        title="Assets and Liabilities Over Time",
        labels={"Amount": "Amount ($)", "Date": "Date", "Metric": "Financial Metric"}
    )

    fig.update_layout(
        xaxis=dict(type="category"),
        yaxis=dict(tickformat=","),
        bargap=0.2
    )

    return fig

def plot_free_cashflow(dataframe):
    """
    Creates scatter plot of yearly or quarterly free cashflow
    
    Parameters:
    - dataframe: Cashflow dataframe

    Returns:
    - Plotly scatter plot of yearly or quarterly free cashflow
    """

    filtered_df = dataframe.loc['FreeCashFlow']
    filtered_df = filtered_df.T.reset_index()
    filtered_df.rename(columns={"index": "Date"}, inplace=True)
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"]).dt.date
    filtered_df = filtered_df.loc[::-1]

    fig = px.line(
    filtered_df, 
    x=filtered_df['Date'], 
    y=filtered_df['FreeCashFlow'],
    title='Free Cash Flow Over Time',
    labels={"FreeCashFlow": "Free Cash Flow ($)"}
    )
    
    fig.update_layout(
        xaxis=dict(type="category"),
        yaxis=dict(tickformat=",")
    )

    return fig

def plot_earnings_history(dataframe):
    """
    Creates earnings plot, displaying Actual EPS and Estimated EPS
    
    Parameters:
    - dataframe: Earnings history dataframe

    Returns:
    - Plotly scatter graph with actual EPS and estimated EPS, with surprise % values
    """

    dataframe = dataframe.reset_index()
    dataframe.rename(columns={"index": "Date"}, inplace=True)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"]).dt.date

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataframe['Date'],
        y=dataframe['epsEstimate'],
        mode='markers',
        marker=dict(size=12, symbol='circle-open', color='blue', line=dict(width=2, color='gray')),
        name='EPS Estimate'
    ))

    for i, row in dataframe.iterrows():
        color = 'green' if row['epsActual'] > row['epsEstimate'] else 'red'
        fig.add_trace(go.Scatter(
            x=[row['Date']],
            y=[row['epsActual']],
            mode='markers',
            marker=dict(size=12, color=color, line=dict(width=1, color='black')),
            name="Actual EPS" if i == 0 else "",  
            hovertemplate=(
                f"<b>EPS Actual</b>: {row['epsActual']:.2f}<br>"
                f"<b>Surprise %</b>: {row['surprisePercent']:.2f}%"
                "<extra></extra>"
            ),
            showlegend=(i == 0)  
        ))

    fig.update_layout(
        title="Quarterly EPS Actual vs Estimate",
        xaxis=dict(title="Date", tickmode='array', tickvals=dataframe['Date']),
        yaxis=dict(title="EPS", zeroline=True, zerolinecolor='gray', zerolinewidth=1),
        font=dict(color='black'),
        legend=dict(x=1, y=1),
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig

def obtain_dataframe(selected_ticker):
    """
    Creates stock dataframe and adds technical indicators to create target and features
    
    Parameters:
    - selected_ticker: Selected ticker symbol

    Returns:
    - Dataframe with technical indicators 
    """

    ticker = yf.Ticker(selected_ticker)
    dataframe = ticker.history(period='max')
    dataframe.drop(columns=['Dividends', 'Stock Splits'], inplace=True)
    dataframe = dataframe.loc['2010-01-01':].copy()
    dataframe['Garman_Klass_Volatility'] = ((np.log(dataframe['High'])-np.log(dataframe['Low']))**2)/2-(2*np.log(2)-1)*((np.log(dataframe['Close'])-np.log(dataframe['Open']))**2)
    dataframe['RSI'] = ta.rsi(close=dataframe['Close'], length=14)
    dataframe['BB_Low'] = ta.bbands(close=dataframe['Close'], length=20).iloc[:,0]
    dataframe['BB_Mid'] = ta.bbands(close=dataframe['Close'], length=20).iloc[:,1]
    dataframe['BB_High'] = ta.bbands(close=dataframe['Close'], length=20).iloc[:,2]
    dataframe['EMA_50'] = ta.ema(dataframe['Close'], length=50)
    dataframe['EMA_200'] = ta.ema(dataframe['Close'], length=200)
    dataframe['MACD_12_26_9'] = ta.macd(dataframe['Close'], fast=12, slow=26, signal=9).iloc[:,0]
    dataframe['ATR'] = ta.atr(dataframe['High'], dataframe['Low'], dataframe['Close'])
    dataframe['ADI'] = ta.ad(dataframe["High"], dataframe["Low"], dataframe["Close"], dataframe["Volume"])
    dataframe['ADX_14'] = ta.adx(dataframe["High"], dataframe["Low"], dataframe["Close"], length=14).iloc[:,2]
    dataframe['OBV'] = ta.obv(dataframe["Close"], dataframe["Volume"])

    return dataframe

def get_features_targets(dataframe):
    """
    Creates target and scaled features for LSTM to generate forecasts
    
    Parameters:
    - dataframe: Dataframe with technical indicators

    Returns:
    - Target and scaled features to be used by model to generate forecasts
    """

    target = pd.DataFrame(dataframe['Close'])
    features = pd.DataFrame(dataframe.drop(columns=['Close']))
    scaler = MinMaxScaler(feature_range=(0, 1))
    feature_transform = scaler.fit_transform(features)
    feature_transform = pd.DataFrame(data=feature_transform, index=dataframe.index)

    return target, feature_transform

def forecast(model, feature_transform, target, forecast_days=30):
    """
    Uses scaled features and pre-trained LSTM model to generate a list of forecasts/predictions of the stock price for the next 30 days
    
    Parameters:
    - model: Pre-trained LSTM model
    - feature_transform: Scaled features
    - target: Target dataframe with close prices
    - forecast_days: Number of days to forecast prices

    Returns:
    - List of forecasted/predicted prices for the next 30 days
    """
    
    # Get the last sequence of features
    last_sequence = feature_transform.iloc[-1:].values
    last_sequence = last_sequence.reshape(1, 1, -1)
    
    # Get the last known close price
    last_close = target.iloc[-1].values[0]
    
    # List to store predictions
    predictions = []
    predictions.append(last_close)
    
    for _ in range(forecast_days - 1):
        # Predict the next day's price
        next_pred = model.predict(last_sequence, verbose=0)
        next_price = next_pred[0, 0]
        predictions.append(next_price)
        
        # Create a new sequence for the next prediction
        # Add some small random variation to prevent static predictions
        new_sequence = last_sequence[0, 0, :].copy()
        
        # Add a small amount of noise to features
        noise = np.random.normal(0, 0.01, new_sequence.shape)
        new_sequence += noise
        
        # Reshape for next prediction
        last_sequence = new_sequence.reshape(1, 1, -1)
    
    return predictions

def create_predictions_series(dataframe, predictions):
    """
    Creates a Pandas series of forecasted prices attached to the dates
    
    Parameters:
    - dataframe: Dataframe with technical indicators
    - predictions: List of forecasted/predicted prices

    Returns:
    - Pandas series of forecasted/predicted prices
    """

    last_date = dataframe.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    predictions_series = pd.Series(predictions, index=future_dates)

    return predictions_series

def plot_forecast(selected_ticker, dataframe, predictions_series):
    """
    Creates forecast plot displaying the historical prices of the last 30 days and the forecasted prices for the next 30 days
    
    Parameters:
    - selected_ticker: Selected ticker symbol
    - dataframe: Dataframe with technical indicators
    - predictions_series: Pandas series of forecasted/predicted prices

    Returns:
    - Target and scaled features to be used by model to generate forecasts
    """

    last_month_data = dataframe[dataframe.index >= (dataframe.index[-1] - pd.DateOffset(months=1))]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=last_month_data.index, y=last_month_data['Close'], mode='lines', name='Historical Prices'))

    fig.add_trace(go.Scatter(x=predictions_series.index, y=predictions_series, mode='lines', name='Predicted Prices', line=dict(color='red')))

    fig.update_layout(
        title=f'{selected_ticker} 30 Day Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        template='plotly_dark'
    )

    return fig

def process_news(selected_ticker):
    """
    Obtains news data for a given ticker
    
    Parameters:
    - selected_ticker: Selected ticker symbol

    Returns:
    - Dictionary of the news data (title, publisher, link, thumbnail)
    """

    ticker = yf.Ticker(selected_ticker)
    news_list = ticker.get_news()

    news_data = {}
    
    for idx, story in enumerate(news_list, start=1):
        thumbnail_resolutions = story.get('thumbnail', {}).get('resolutions', [])
        first_thumbnail_url = thumbnail_resolutions[0]['url'] if thumbnail_resolutions else None    

        news_data[f"story_{idx}"] = {
            "title": story.get('title'),
            "publisher": story.get('publisher'),
            "link": story.get('link'),
            "thumbnail": first_thumbnail_url,
        }
    
    return news_data