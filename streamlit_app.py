import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_URL = "https://raw.githubusercontent.com/JamesBustamante44/Airline-Passenger-Satisfaction-Classification-Project/main/cleaned_airline_passenger_satisfaction.csv"
RANDOM_STATE = 42

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    if 'Arrival Delay in Minutes' in df.columns:
        mean_delay = df['Arrival Delay in Minutes'].mean()
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(mean_delay)
    df['satisfaction_encoded'] = df['satisfaction'].apply(lambda x: 1 if str(x).lower() == 'satisfied' else 0)
    return df

df = load_data()
if df is None:
    st.stop()

st.title('Airline Passenger Satisfaction Classification Project')


def get_feature_sets(dataframe):
    X = dataframe.drop(columns=['satisfaction', 'satisfaction_encoded'])
    cat = X.select_dtypes(include=['object']).columns.tolist()
    num = X.select_dtypes(include=['number']).columns.tolist()
    return X, cat, num


X_full, CATEGORICAL_FEATURES, NUMERIC_FEATURES = get_feature_sets(df)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), NUMERIC_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
    ]
)

@st.cache_resource
def train_models(data):
    X = data.drop(columns=['satisfaction', 'satisfaction_encoded'])
    y = data['satisfaction_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=RANDOM_STATE),
        'Random Forest Classifier': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        # 'Support Vector Machine (SVC)': SVC(random_state=RANDOM_STATE, probability=True)
    }

    trained = {}
    results = {}
    for name, model in models.items():
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['dissatisfied', 'satisfied'], output_dict=True)
        trained[name] = pipe
        results[name] = {'accuracy': acc, 'report': report, 'y_test': y_test, 'y_pred': y_pred}
    return trained, results

with st.spinner('Training models (cached)...'):
    trained_models, model_results = train_models(df)

# Tabs
tab_overview, tab_eda, tab_modeling, tab_predict = st.tabs(['Overview', 'EDA', 'Modeling', 'Predict'])

with tab_overview:
    st.header('Dataset Overview')
    st.write(f'Shape: {df.shape[0]} rows × {df.shape[1]} columns')
    st.subheader('Head')
    st.dataframe(df.head())
    st.subheader('Dtypes & Nulls')
    info_df = pd.DataFrame({'dtype': df.dtypes, 'non_null': df.count(), 'nulls': df.isnull().sum()})
    st.dataframe(info_df)

with tab_eda:
    st.header('Exploratory Data Analysis')
    st.subheader('Categorical Feature Distribution')
    cat_choices = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
    cat_feat = st.selectbox('Pick a categorical feature:', cat_choices)
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x=cat_feat, ax=ax1, palette='viridis')
    ax1.set_title(f'Count of {cat_feat}')
    ax1.tick_params(axis='x', rotation=30)
    st.pyplot(fig1)

    st.subheader('Numeric Feature Distribution')
    num_choices = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
    num_feat = st.selectbox('Pick a numeric feature:', num_choices)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.histplot(df[num_feat], kde=True, ax=ax2, color='steelblue')
    ax2.set_title(f'Histogram of {num_feat}')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 2))
    sns.boxplot(x=df[num_feat], ax=ax3, color='salmon')
    ax3.set_title(f'Boxplot of {num_feat}')
    st.pyplot(fig3)

    st.subheader('Class vs Satisfaction (Stacked)')
    ct = pd.crosstab(df['Class'], df['satisfaction'], normalize='index') * 100
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ct.plot(kind='bar', stacked=True, ax=ax4, colormap='coolwarm')
    ax4.set_ylabel('Percentage')
    ax4.set_title('Satisfaction by Class')
    ax4.tick_params(axis='x', rotation=0)
    st.pyplot(fig4)

with tab_modeling:
    st.header('Model Training and Evaluation')
    name = st.selectbox('Select a model:', list(trained_models.keys()))
    res = model_results[name]
    st.metric('Accuracy', f"{res['accuracy']:.4f}")
    st.subheader('Classification Report')
    st.dataframe(pd.DataFrame(res['report']).transpose())
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(res['y_test'], res['y_pred'])
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Dissatisfied (0)', 'Satisfied (1)'],
                yticklabels=['Dissatisfied (0)', 'Satisfied (1)'],
                ax=ax_cm)
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('True')
    ax_cm.set_title(f'Confusion Matrix · {name}')
    st.pyplot(fig_cm)

with tab_predict:
    st.header('Predict Passenger Satisfaction')
    name = st.selectbox('Model for prediction:', list(trained_models.keys()), key='pred_model')
    model = trained_models[name]

    X_current, cat_current, num_current = get_feature_sets(df)
    st.subheader('Input Features')
    with st.form('predict_form'):
        cols = st.columns(3)
        data = {}
        features = X_current.columns.tolist()
        for i, feat in enumerate(features):
            col = cols[i % 3]
            with col:
                if feat in num_current:
                    if pd.api.types.is_float_dtype(df[feat]):
                        vmin = float(df[feat].min())
                        vmax = float(df[feat].max())
                        default = float(df[feat].mean())
                        step = 1.0
                        fmt = '%f'
                    else:
                        vmin = int(df[feat].min())
                        vmax = int(df[feat].max())
                        default = int(df[feat].mean())
                        step = 1
                        fmt = '%d'
                    data[feat] = st.number_input(feat, value=default, min_value=vmin, max_value=vmax, step=step, format=fmt)                
            else:
                options = df[feat].dropna().unique().tolist()
                                if not options:
                                    options = ['']
                    data[feat] = st.selectbox(feat, options)
        submitted = st.form_submit_button('Predict')

    if submitted:
        try:
            inp = pd.DataFrame([data])[features]
            pred = model.predict(inp)[0]
            label = 'Satisfied' if pred == 1 else 'Dissatisfied'
            st.success(f'Predicted: {label}')
            if hasattr(model.named_steps['classifier'], 'predict_proba'):
                proba = model.predict_proba(inp)[0]
                st.write(pd.DataFrame({'Class': ['Dissatisfied', 'Satisfied'], 'Probability': proba}).set_index('Class'))
            elif hasattr(model.named_steps['classifier'], 'decision_function'):
                st.info('Selected model provides decision scores instead of probabilities.')
        except Exception as e:
            st.error(f'Prediction error: {e}')
