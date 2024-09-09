#!/usr/bin/env python
# coding: utf-8

# In[11]:


#!/usr/bin/env python
# coding: utf-8

# Imports
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from pandas.tseries.offsets import BDay

# Configure the page
st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
)

# Load the LSTM model
model = load_model('lstm_combined_model.keras')

# Load the scalers
scaler_X = joblib.load('scaler_X.pkl')  # Load the scaler for features
scaler_y = joblib.load('scaler_y.pkl')  # Load the scaler for target variable

# Load the data from the CSV file
data = pd.read_csv("stock_technical_indicators_last_5_years.csv")
data.dropna(inplace=True)

# Ensure the 'date' column is in datetime format
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Sidebar configuration
st.sidebar.markdown("## **User Input Features**")

# Unique tickers
tickers = data['ticker'].unique()
ticker_dict = {ticker: idx for idx, ticker in enumerate(tickers)}

# Add a dropdown for selecting the stock
st.sidebar.markdown("### **Select stock**")
stock = st.sidebar.selectbox("Choose a stock", tickers)

# Add a selector for forecast days
st.sidebar.markdown("### **Select forecast days**")
forecast_days = st.sidebar.slider("Choose number of forecast days", min_value=1, max_value=30, value=10)

# Add a disabled input for stock ticker
st.sidebar.markdown("### **Stock ticker code**")
st.sidebar.text_input(label="Ticker code", placeholder=stock, disabled=True)

# Add title to the app
st.markdown("# **Stock Price Prediction**")
st.markdown("##### **Enhance Investment Decisions through Data-Driven Forecasting**")

# Prepare data for the selected stock
ticker_idx = ticker_dict[stock]
ticker_data = data[data['ticker'] == stock]

# Debugging: Check if ticker_data is empty
if ticker_data.empty:
    st.error("No data available for the selected stock.")
else:
    X_ticker = ticker_data[['EMA', 'RSI']].values
    
    # Check if X_ticker has data
    if X_ticker.shape[0] == 0:
        st.error("No 'EMA' and 'RSI' data available for the selected stock.")
    else:
        # Normalize the data
        X_ticker_scaled = scaler_X.transform(X_ticker)
        X_ticker_scaled = np.reshape(X_ticker_scaled, (X_ticker_scaled.shape[0], 1, X_ticker_scaled.shape[1]))

        # Predict using the model
        predictions = model.predict(X_ticker_scaled)
        predictions = scaler_y.inverse_transform(predictions)

        # Forecast future values
        future_predictions = []
        last_sequence = X_ticker_scaled[-1]

        for _ in range(forecast_days):
            next_prediction = model.predict(np.array([last_sequence]))
            future_predictions.append(next_prediction[0])
            last_sequence = np.append(last_sequence[:, 1:], next_prediction, axis=1)

        future_predictions = scaler_y.inverse_transform(future_predictions)

        # Convert future_predictions to a NumPy array and flatten it
        future_predictions = np.array(future_predictions).flatten()

        # Adjust the future predictions to smooth the forecast
        smoothed_predictions = []
        for i in range(len(future_predictions)):
            if i > 0:
                # Reduce the value if it's more than 20% higher than the previous value
                if future_predictions[i] > 1.2 * smoothed_predictions[-1]:
                    smoothed_predictions.append(smoothed_predictions[-1] * 1.1)
                # Increase the value if it's more than 20% lower than the previous value
                elif future_predictions[i] < 0.8 * smoothed_predictions[-1]:
                    smoothed_predictions.append(smoothed_predictions[-1] * 0.9)
                else:
                    smoothed_predictions.append(future_predictions[i])
            else:
                smoothed_predictions.append(future_predictions[i])
        
        future_predictions = np.array(smoothed_predictions)

        # Function to get the next business day
        def get_next_business_day(start_date):
            next_business_day = start_date + BDay(1)
            return next_business_day

        # Current date and time
        current_date = pd.Timestamp.now()

        # Get the next business day
        next_business_day = get_next_business_day(current_date)

        # Create a DataFrame for the forecast starting from the next business day
        forecast_index = pd.date_range(start=next_business_day, periods=forecast_days, freq='B')
        forecast_df = pd.DataFrame({
            'Date': forecast_index,
            'Forecasted Price': future_predictions
        })

        # Determine buy/sell signals
        def determine_signals(data, include_forecast=False):
            if include_forecast:
                data['EMA_short'] = data['Forecasted Price'].ewm(span=12, adjust=False).mean()
                data['EMA_long'] = data['Forecasted Price'].ewm(span=26, adjust=False).mean()
            else:
                # Ensure EMA columns are available
                if 'EMA_short' not in data.columns or 'EMA_long' not in data.columns:
                    data['EMA_short'] = data['c'].ewm(span=12, adjust=False).mean()  # Short-term EMA
                    data['EMA_long'] = data['c'].ewm(span=26, adjust=False).mean()  # Long-term EMA

            # Define a threshold for approximate equality
            threshold = 0.005 * data['EMA_long']  # 0.5% of the long-term EMA

            # Determine the signals based on EMAs and the threshold
            data['Signal'] = np.where(
                (data['EMA_short'] > data['EMA_long'] + threshold),
                'Buy Stocks',
                np.where(
                    (data['EMA_short'] < data['EMA_long'] - threshold),
                    'Sell Stocks',
                    np.where(
                        (np.abs(data['EMA_short'] - data['EMA_long']) <= threshold),
                        'Consider Entry/Hold Stocks',
                        'Hold Stocks/Consider Entry'
                        
                    )
                )
            )
            return data

        ticker_data = determine_signals(ticker_data)

        # Historical Data Graph
        st.markdown("## **Historical Data**")
        hover_text = [
            f"Date: {date}<br>High: {high}<br>Low: {low}<br>Close: {close}"
            for date, high, low, close in zip(ticker_data.index, ticker_data["h"], ticker_data["l"], ticker_data["c"])
        ]
        fig = go.Figure(data=[
            go.Scatter(
                x=ticker_data.index, y=ticker_data["c"], mode="lines", name="Closing Price",
                text=hover_text, hoverinfo='text'
            )
        ])
        fig.update_layout(title=f"Historical Closing Prices for {stock}", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Stock Prediction Graph
        st.markdown("## **Stock Prediction**")
        predictions_series = pd.Series(predictions.flatten(), index=ticker_data.index[:len(predictions)])
        
        # Create prediction graph
        fig = go.Figure(data=[
            go.Scatter(x=ticker_data.index, y=ticker_data["c"], name="Actual", mode="lines", line=dict(color="blue")),
            go.Scatter(x=ticker_data.index, y=predictions_series, name="Predictions", mode="lines", line=dict(color="red")),
            go.Scatter(x=forecast_index, y=future_predictions, name="Future Forecast", mode="lines", line=dict(color="green")),
        ])

        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Display forecasted values in table format
        forecast_df['RSI'] = [data['RSI'].iloc[-1]] * forecast_days  # Use the last RSI value for simplicity
        forecast_df = determine_signals(forecast_df, include_forecast=True)
        
        st.markdown(f"### **Forecasted values for the next {forecast_days} business days:**")
        styled_forecast_df = forecast_df.style.applymap(
            lambda x: 'background-color: green' if isinstance(x, str) and 'Buy' in x else 'background-color: red' if isinstance(x, str) and 'Sell' in x else '',
            subset=['Signal']
        ).format({"Forecasted Price": "${:.2f}"})
        st.dataframe(styled_forecast_df)
        
        # Download forecasted values as CSV
        st.markdown("### **Download Forecasted Values**")
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f'forecast_{stock}.csv',
            mime='text/csv'
        )

