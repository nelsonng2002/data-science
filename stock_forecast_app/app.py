import streamlit as st
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
from functions import get_sp500_tickers
from functions import df_for_candlestick, plot_candlestick
from functions import create_stats_dataframe
from functions import income_statement, balance_sheet, cashflow, earnings_history
from functions import calculate_current_ratio
from functions import plot_income_statement, plot_assets_liabilities, plot_free_cashflow, plot_earnings_history
from functions import obtain_dataframe
from functions import get_features_targets, forecast, create_predictions_series, plot_forecast
from functions import process_news

pd.options.display.float_format = '{:.2f}'.format

stock_keys = [
'previousClose', 'open', 'dayLow', 'dayHigh', 'volume', 'averageVolume', 'averageVolume10days',
'averageDailyVolume10Day', 'marketCap', 'fiftyTwoWeekLow', 'fiftyTwoWeekHigh', 
'fiftyDayAverage', 'twoHundredDayAverage',
'52WeekChange', 'SandP52WeekChange', 'priceToSalesTrailing12Months', 
]

income_keys = [
    'totalRevenue', 'revenuePerShare', 'profitMargins', 'grossMargins', 'ebitdaMargins', 
    'operatingMargins', 'ebitda', 'trailingPE', 'forwardPE', 
    'trailingEps', 'forwardEps', 
    'enterpriseToRevenue', 'enterpriseToEbitda', 'trailingPegRatio'
'earningsQuarterlyGrowth', 'earningsGrowth', 'revenueGrowth', 
]

balance_keys = [
    'totalCash', 'totalDebt', 'quickRatio', 
    'currentRatio', 'debtToEquity',
    'bookValue', 'priceToBook', 'totalCashPerShare',  
    'returnOnAssets', 'returnOnEquity'
]

cashflow_keys = [
    'freeCashflow', 'operatingCashflow'
]

share_keys = [
    'floatShares', 'sharesOutstanding', 'sharesShort', 'sharesPercentSharesOut',
    'heldPercentInsiders', 'heldPercentInstitutions', 'shortRatio',
    'shortPercentOfFloat', 'impliedSharesOutstanding'
]

dividend_splits_keys = [
    'dividendRate', 'dividendYield', 'exDividendDate', 'payoutRatio',
    'fiveYearAvgDividendYield', 'trailingAnnualDividendRate', 'trailingAnnualDividendYield',
    'lastSplitFactor', 'lastSplitDate'
]

sidebar_text = '''
<div style="background-color: #9c2230; padding: 20px; border-radius: 10px; text-align: center;">
    <h1 style="color: #fff; font-size: 24px;">Stock Price Data, Statistics and Forecast App</h1>
    <p style="color: #fff; font-size: 16px;">A LSTM Model</p>
</div>

## About This App

This application allows users to visualise stock price data, statistics and financial data. Furthermore, the application leverages a pre-trained LSTM model to forecast stock prices for the next 30 days.
The LSTM model was trained and developed using Tensorflow Keras. The stock data is obtained using the Yahoo Finance Python API. 

### Disclaimer

- The information provided is for educational and demonstration purposes only.
- This is not financial advice.

Learn more about the app
'''

st.sidebar.markdown(sidebar_text, unsafe_allow_html=True)

st.title('Stock Data, Statistics and Forecasting 📊💸')

introduction = '''
To get started, just choose a stock ticker!
'''
st.markdown(introduction)

# Ticker Selection --------------------------------------------------------------------------------------------------------------------------------
tickers = get_sp500_tickers()
selected_ticker = st.selectbox('Select a stock ticker', tickers)

# Candlestick Chart --------------------------------------------------------------------------------------------------------------------------------
selection_horizon = ''

st.markdown('Choose a horizon for the stock price candlestick chart 📈')
one_day, five_day, one_month, six_month, one_year, five_year = st.columns(6)
if one_day.button("1d", use_container_width=True):
    selected_horizon='1d'
    candlestick_dataframe = df_for_candlestick(selected_horizon, selected_ticker)
    candlestick = plot_candlestick(selected_ticker, selected_horizon, candlestick_dataframe)
    st.plotly_chart(candlestick)
if five_day.button("5d", use_container_width=True):
    selected_horizon='5d'
    candlestick_dataframe = df_for_candlestick(selected_horizon, selected_ticker)
    candlestick = plot_candlestick(selected_ticker, selected_horizon, candlestick_dataframe)
    st.plotly_chart(candlestick)
if one_month.button("1mo", use_container_width=True):
    selected_horizon='1mo'
    candlestick_dataframe = df_for_candlestick(selected_horizon, selected_ticker)
    candlestick = plot_candlestick(selected_ticker, selected_horizon, candlestick_dataframe)
    st.plotly_chart(candlestick)
if six_month.button("6mo", use_container_width=True):
    selected_horizon='6mo'
    candlestick_dataframe = df_for_candlestick(selected_horizon, selected_ticker)
    candlestick = plot_candlestick(selected_ticker, selected_horizon, candlestick_dataframe)
    st.plotly_chart(candlestick)
if one_year.button("1y", use_container_width=True):
    selected_horizon='1y'
    candlestick_dataframe = df_for_candlestick(selected_horizon, selected_ticker)
    candlestick = plot_candlestick(selected_ticker, selected_horizon, candlestick_dataframe)
    st.plotly_chart(candlestick)
if five_year.button("5y", use_container_width=True):
    selected_horizon='5y'
    candlestick_dataframe = df_for_candlestick(selected_horizon, selected_ticker)
    candlestick = plot_candlestick(selected_ticker, selected_horizon, candlestick_dataframe)
    st.plotly_chart(candlestick)

# Statistics --------------------------------------------------------------------------------------------------------------------------------
stock_stats = create_stats_dataframe(selected_ticker, stock_keys, 'Stock')
income_stats = create_stats_dataframe(selected_ticker, income_keys, 'Income Statement')
balance_stats = create_stats_dataframe(selected_ticker, balance_keys, 'Balance Sheet')
cashflow_stats = create_stats_dataframe(selected_ticker, cashflow_keys, 'Cash Flow')
share_stats = create_stats_dataframe(selected_ticker, share_keys, 'Share')
dividend_splits_stats = create_stats_dataframe(selected_ticker, dividend_splits_keys, 'Dividends and Splits')

st.header("Statistics")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Stock Statistics", "Income Statement Statistics", "Balance Sheet Statistics", "Cash Flow Statistics", "Share Statistics", "Dividends and Splits Statistics"])

with tab1:
    st.table(stock_stats)
with tab2:
    st.table(income_stats)
with tab3:
    st.table(balance_stats)
with tab4:
    st.table(cashflow_stats)
with tab5:
    st.table(share_stats)
with tab6:
    st.table(dividend_splits_stats)

# Financial Information --------------------------------------------------------------------------------------------------------------------------------
st.header("Financial Information")
selected_financial_timeframe = st.selectbox('Select yearly or quarterly financial information', options=['yearly', 'quarterly'])

tab1, tab2, tab3 = st.tabs(["Income Statement", "Balance Sheet", "Cash Flow"])
with tab1:
    income_statement_df = income_statement(selected_ticker, selected_financial_timeframe)
    st.header("Income Statement")
    st.table(income_statement_df)
    income_stmt_bar = plot_income_statement(income_statement_df)
    st.plotly_chart(income_stmt_bar)
with tab2:
    balance_sheet_df = balance_sheet(selected_ticker, selected_financial_timeframe)
    st.header("Balance Sheet")
    st.table(balance_sheet_df)
    total_assets_liabilities_plot = plot_assets_liabilities(balance_sheet_df)
    st.plotly_chart(total_assets_liabilities_plot)
    current_ratio = calculate_current_ratio(balance_sheet_df)
    st.markdown(f"Current Ratio = {current_ratio}")

with tab3:
    cashflow_df = cashflow(selected_ticker, selected_financial_timeframe)
    st.header("Cash Flow")
    st.table(cashflow_df)
    free_cashflow_line = plot_free_cashflow(cashflow_df)
    st.plotly_chart(free_cashflow_line)

# Earnings --------------------------------------------------------------------------------------------------------------------------------
st.header("Earnings History")
earnings_history_df = earnings_history(selected_ticker)
st.dataframe(earnings_history_df)

earnings_history_plot = plot_earnings_history(earnings_history_df)
st.plotly_chart(earnings_history_plot)

# Forecast --------------------------------------------------------------------------------------------------------------------------------
forecast_dataframe = obtain_dataframe(selected_ticker)
target, feature_transform = get_features_targets(forecast_dataframe)
forecast_model = load_model('forecast_model.h5')
predictions = forecast(forecast_model, feature_transform, target, forecast_days=30)
predictions_series = create_predictions_series(forecast_dataframe, predictions)
forecast_plot = plot_forecast(selected_ticker, forecast_dataframe, predictions_series)

st.header("Stock Price Forecast")
forecast_desc = '''
This plot displays the forecasted prices of the selected stock for the next 30 days. This forecasting is executed by a LSTM model using the stock's historical price data as well as technical indicators.
'''
st.markdown(forecast_desc)
st.plotly_chart(forecast_plot)

# News Feed --------------------------------------------------------------------------------------------------------------------------------
st.header(f'{selected_ticker} News Feed')
news_stories = process_news(selected_ticker)

col1, col2 = st.columns(2)
with col1:
    st.subheader(news_stories['story_1']['title'])
    st.caption(news_stories['story_1']['publisher'])
    if news_stories['story_1']['thumbnail']:
        st.image(news_stories['story_1']['thumbnail'])
    link1 = news_stories['story_1']['link']
    st.markdown('Read the story [here](%s)' % link1)
with col2:
    st.subheader(news_stories['story_2']['title'])
    st.caption(news_stories['story_2']['publisher'])
    if news_stories['story_2']['thumbnail']:
        st.image(news_stories['story_2']['thumbnail'])
    link2 = news_stories['story_2']['link']
    st.markdown('Read the story [here](%s)' % link2)

col3, col4 = st.columns(2)
with col3:
    st.subheader(news_stories['story_3']['title'])
    st.caption(news_stories['story_3']['publisher'])
    if news_stories['story_3']['thumbnail']:
        st.image(news_stories['story_3']['thumbnail'])
    link3 = news_stories['story_3']['link']
    st.markdown('Read the story [here](%s)' % link3)
with col4:
    st.subheader(news_stories['story_4']['title'])
    st.caption(news_stories['story_4']['publisher'])
    if news_stories['story_4']['thumbnail']:
        st.image(news_stories['story_4']['thumbnail'])
    link4 = news_stories['story_4']['link']
    st.markdown('Read the story [here](%s)' % link4)

col5, col6 = st.columns(2)
with col5:
    st.subheader(news_stories['story_5']['title'])
    st.caption(news_stories['story_5']['publisher'])
    if news_stories['story_5']['thumbnail']:
        st.image(news_stories['story_5']['thumbnail'])
    link5 = news_stories['story_5']['link']
    st.markdown('Read the story [here](%s)' % link5)
with col6:
    st.subheader(news_stories['story_6']['title'])
    st.caption(news_stories['story_6']['publisher'])
    if news_stories['story_6']['thumbnail']:
        st.image(news_stories['story_6']['thumbnail'])
    link6 = news_stories['story_6']['link']
    st.markdown('Read the story [here](%s)' % link6)

