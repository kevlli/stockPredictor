import streamlit as sl
from datetime import date
import yfinance as yf
from pytickersymbols import PyTickerSymbols
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go


TODAY = date.today().strftime("%Y-%m-%d")
sl.title("Stock Predictor")

START = sl.date_input("Start Date for stock analysis", max_value=date.today(), value = date(2018, 1, 1))
stock_data = PyTickerSymbols()
tickers = stock_data.get_sp_100_nyc_yahoo_tickers()
#print(tickers)
selected_stock = sl.selectbox("Select the stock", tickers)

n_years = sl.slider("Years to predict:", 1.0, 5.0, step = 0.1)
p = int(n_years * 365)

def load_data(stock):
    data = yf.download(stock, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)

sl.subheader("Stock data")
sl.write(data.tail())

figl = go.Figure()
figl.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open_price"))
figl.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close_price"))
figl.layout.update(title_text="Stock Price Graph", xaxis_rangeslider_visible = True)
sl.plotly_chart(figl)

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns = {"Date": "ds", "Close" : "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=p)
test = m.predict(future)

sl.subheader("Predictive data")
sl.write(test.tail())
fig2 = plot_plotly(m, test)
sl.plotly_chart(fig2)