import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from prophet import Prophet

# Set page title
st.set_page_config(page_title="Database Energy Forecast", layout="wide")
st.title("🔮 Database Energy Consumption Forecast")
st.markdown("This app forecasts the next week's energy usage based on historical data.")

# Load the trained model
@st.cache_resource
def load_model():
    with open('prophet_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Sidebar: user controls
st.sidebar.header("Forecast Options")
days = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=14, value=7, step=1)
periods = days * 24  # hours

# Generate forecast
future = model.make_future_dataframe(periods=periods, freq='H')
forecast = model.predict(future)

# Extract future part (only the forecasted period)
last_historical = forecast[forecast['ds'] <= forecast['ds'].max() - pd.Timedelta(days=days)].iloc[-1]['ds']
future_forecast = forecast[forecast['ds'] > last_historical].copy()

# --- Main content: two columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Forecast Plot")
    fig = model.plot(forecast, uncertainty=True)
    st.pyplot(fig)
    # Also show a smaller plot of the forecasted part only
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.plot(future_forecast['ds'], future_forecast['yhat'], label='Forecast', color='blue')
    ax.fill_between(future_forecast['ds'],
                    future_forecast['yhat_lower'],
                    future_forecast['yhat_upper'],
                    alpha=0.2, color='blue', label='Uncertainty')
    ax.set_title(f"Next {days} Days Energy Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy (Joules)")
    ax.legend()
    st.pyplot(fig2)

with col2:
    st.subheader("📊 Forecast Data")
    # Show forecast table
    st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(24))
    st.caption(f"Showing first 24 hours of the forecast. Total rows: {len(future_forecast)}")

# Optional: show model components
if st.checkbox("Show model components (trend, seasonality)"):
    fig3 = model.plot_components(forecast)
    st.pyplot(fig3)

# Optional: download forecast as CSV
csv = future_forecast.to_csv(index=False)
st.download_button("Download forecast as CSV", csv, file_name="energy_forecast.csv", mime="text/csv")