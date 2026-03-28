# Database Energy Efficiency Forecast

This project uses a **Prophet** time series model to forecast next week's energy consumption of a database system, based on historical query logs and energy measurements (synthetic data for demonstration).

## How to run the app

1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Files

- `app.py` – Streamlit dashboard
- `prophet_model.pkl` – Trained Prophet model
- `hourly_energy.csv` – Historical hourly energy data (optional)
- `requirements.txt` – Python dependencies

## Model training

The model was trained on synthetic data that mimics database queries and their energy consumption. See the notebook `train_model.ipynb` for details.