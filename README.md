# Airline Passenger Satisfaction

## Quickstart

- Create a virtual env (optional)
- Install deps: `pip install streamlit scikit-learn pandas seaborn matplotlib`
- Generate cleaned CSV: `python scripts/generate_cleaned_csv.py`
- Run the app: `streamlit run app/streamlit_app.py`

## Notebooks

- notebooks/EDA.ipynb — loads the raw CSV, applies basic cleaning, and can export data/cleaned_airline_passenger_satisfaction.csv.
- notebooks/modeling.ipynb — prefers the cleaned CSV if present; otherwise pulls raw and applies the same cleaning, then trains LR/RF/SVC.

Run EDA first if you want to materialize the cleaned CSV locally.
