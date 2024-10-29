import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv("Unemployment in India.csv")  # Replace with your local data path
df.columns = df.columns.str.strip()  # Remove spaces from column names
df.rename(columns={
    'Estimated Unemployment Rate (%)': 'Unemployment Rate',
    'Estimated Employed': 'Employed',
    'Estimated Labour Participation Rate (%)': 'Labour Participation Rate',
    'Area': 'Area'
}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')  # Convert Date column to datetime
df = df.dropna()  # Remove rows with missing values

# Sidebar
st.sidebar.title("Filters")
selected_region = st.sidebar.selectbox("Select Region", df['Region'].unique())
start_date, end_date = st.sidebar.date_input("Select Date Range", [df['Date'].min().date(), df['Date'].max().date()])

# Convert date inputs to datetime format
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Filter Data
df_filtered = df[(df['Region'] == selected_region) & (df['Date'].between(start_date, end_date))]

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualization", "Forecasting"])

# Tab 1: Data Overview
with tab1:
    st.header("Data Overview")
    st.write("Displaying data for:", selected_region)
    st.write(df_filtered.head())

# Tab 2: Visualization
with tab2:
    st.header("Unemployment Rate Over Time")

    # Moving average calculation
    df_filtered['Unemployment_MA'] = df_filtered['Unemployment Rate'].rolling(window=12).mean()

    # Plot
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df_filtered, x='Date', y='Unemployment Rate', label='Unemployment Rate')
    sns.lineplot(data=df_filtered, x='Date', y='Unemployment_MA', label='12-Month Moving Average')
    plt.title(f"Unemployment Rate and Moving Average in {selected_region}")
    plt.legend()
    st.pyplot(plt)

# Tab 3: Forecasting
with tab3:
    st.header("Forecasting")
    forecast_model = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA"])

    if forecast_model == "Prophet":
        st.subheader("Forecasting with Prophet")
        df_prophet = df_filtered[['Date', 'Unemployment Rate']].rename(columns={'Date': 'ds', 'Unemployment Rate': 'y'})
        model = Prophet()
        model.fit(df_prophet)
        forecast = model.predict(df_prophet)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

    elif forecast_model == "ARIMA":
        st.subheader("Forecasting with ARIMA")

        # Normalize for ARIMA model
        scaler = MinMaxScaler()
        df_filtered['Unemployment Rate'] = scaler.fit_transform(df_filtered[['Unemployment Rate']])

        model_arima = ARIMA(df_filtered['Unemployment Rate'], order=(5, 1, 0))
        model_fit_arima = model_arima.fit()
        forecast_arima = model_fit_arima.forecast(steps=12)

        st.write("ARIMA Forecast:")
        st.write(forecast_arima)
