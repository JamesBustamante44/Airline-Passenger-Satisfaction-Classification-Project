
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Airline Satisfaction', layout='wide')
st.title('Airline Passenger Satisfaction Explorer')

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_airline_passenger_satisfaction.csv', encoding='ascii')

@st.cache_resource
def load_model():
    model = joblib.load('best_model.joblib')
    meta = joblib.load('model_meta.joblib')
    return model, meta

df = load_data()
st.write('Data sample:')
st.dataframe(df.head())

model, meta = load_model()

st.subheader('Quick Predict')
# Build inputs dynamically for a single-row prediction
inputs = {}
for c in meta['num_cols']:
    min_v = float(df[c].min()) if c in df.columns else 0.0
    max_v = float(df[c].max()) if c in df.columns else 1.0
    default_v = float(df[c].median()) if c in df.columns else 0.0
    inputs[c] = st.number_input(c, value=default_v, min_value=min_v, max_value=max_v)

for c in meta['cat_cols']:
    options = df[c].dropna().astype(str).unique().tolist() if c in df.columns else []
    default_opt = options[0] if len(options) > 0 else ''
    inputs[c] = st.selectbox(c, options=options, index=0 if len(options) > 0 else None)

if st.button('Predict satisfaction probability'):
    X_in = pd.DataFrame([inputs])
    proba = model.predict_proba(X_in)[:,1][0]
    st.metric('Predicted probability of satisfied', round(float(proba), 4))
