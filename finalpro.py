import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="PSX Stock Predictor", layout="wide")
st.title("📊 PSX Stock Prediction Dashboard")

ticker = st.text_input("Enter PSX Ticker (Example: HBL.KA)", "HBL.KA")

if st.button("Run Analysis"):

    # ----------------------------
    # Download stock data
    # ----------------------------
    data = yf.download(ticker, start="2020-01-01")

    if data.empty:
        st.error("No data found. Please check the ticker symbol.")
        st.stop()

    # Fix multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure numeric
    for col in ['Open','High','Low','Close','Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()

    # ----------------------------
    # Machine Learning Prediction
    # ----------------------------
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    X = data[['Open','High','Low','Close','Volume']][:-1]
    y = data['Target'][:-1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    st.subheader("Model Accuracy")
    st.write(round(accuracy * 100, 2), "%")

    # ----------------------------
    # Next Day Prediction
    # ----------------------------
    last_row = data[['Open','High','Low','Close','Volume']].iloc[-1:]
    prediction = model.predict(last_row)
    prob = model.predict_proba(last_row)

    st.subheader("Next Day Prediction")

    if prediction[0] == 1:
        st.success("Stock likely to go UP 📈")
    else:
        st.error("Stock likely to go DOWN 📉")

    st.write("UP Probability:", round(prob[0][1] * 100, 2), "%")
    st.write("DOWN Probability:", round(prob[0][0] * 100, 2), "%")

    # ----------------------------
    # Stock Statistics
    # ----------------------------
    st.subheader("Stock Statistics")

    current_price = data['Close'].iloc[-1]
    highest = data['High'].max()
    lowest = data['Low'].min()
    avg_volume = data['Volume'].mean()
    change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Current Price", round(current_price, 2))
    col2.metric("Daily Change %", round(change, 2))
    col3.metric("Highest Price", round(highest, 2))
    col4.metric("Lowest Price", round(lowest, 2))
    col5.metric("Avg Volume", int(avg_volume))

    # ----------------------------
    # Clean Price Chart (No MA)
    # ----------------------------
    st.subheader("Stock Price Chart (Last 200 Days)")

    data_recent = data.tail(200)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(data_recent['Close'], label='Close Price', linewidth=2)

    ax.set_title(f"{ticker} Close Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)