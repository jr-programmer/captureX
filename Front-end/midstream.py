import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

# Set up the Streamlit app title and sidebar
st.title("Southern Area Oil Operations - Data Analytics Platform")
st.sidebar.header("Select Analytics Type")

# Simulated dataset including process, mechanical, and performance data
@st.cache
def load_data():
    data = pd.DataFrame({
        'flow': np.random.normal(100, 10, 1000),  # Process Model
        'level': np.random.normal(80, 8, 1000),   # Process Model
        'pressure': np.random.normal(150, 15, 1000),  # Process Model
        'temperature': np.random.normal(200, 20, 1000),  # Process Model
        'vibration': np.random.normal(0.5, 0.1, 1000),  # Mechanical Model
        'bearing_temp': np.random.normal(50, 5, 1000),  # Mechanical Model
        'lube_oil_temp': np.random.normal(60, 5, 1000),  # Mechanical Model
        'surge_limit': np.random.normal(0.8, 0.05, 1000),  # Mechanical Model
        'pump_performance': np.random.normal(75, 7, 1000),  # Performance Model
        'compressor_curve': np.random.normal(0.9, 0.05, 1000),  # Performance Model
        'failure_risk': np.random.normal(0.5, 0.2, 1000)  # Target for Predictive Analytics
    })
    return data

data = load_data()

# 1. Descriptive Analytics: Equipment Historical Data Overview
def descriptive_analytics(data):
    st.subheader("Descriptive Analytics: Equipment Historical Data")
    st.write(data.describe())  # Display statistical summary

    # Visualize key features in dataset
    st.write("Data Visualization")
    sns.pairplot(data)
    st.pyplot(plt)

# 2. Diagnostic Analytics: Root Cause Analysis using PCA and Drill Down
def diagnostic_analytics(data):
    st.subheader("Diagnostic Analytics: Root Cause Analysis")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.drop('failure_risk', axis=1))  # Exclude target column

    st.write("Explained Variance by PCA Components:", pca.explained_variance_ratio_)

    # Scatter plot for PCA analysis
    fig, ax = plt.subplots()
    ax.scatter(pca_result[:, 0], pca_result[:, 1], c=data['failure_risk'], cmap='viridis')
    ax.set_title("PCA - Diagnostic Analysis")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    st.pyplot(fig)

# 3. Predictive Analytics: Digital Twin and Predict Asset Failures
def predictive_analytics(data, target_column):
    st.subheader("Predictive Analytics: Predict Asset Failures using Digital Twin")

    # Separate features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set and calculate error
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    st.write(f"Mean Squared Error (MSE): {mse}")

    # Visualize prediction vs actual
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    ax.set_xlabel("Actual Failure Risk")
    ax.set_ylabel("Predicted Failure Risk")
    ax.set_title("Prediction vs Actual")
    st.pyplot(fig)

    return model

# 4. Prescriptive Analytics: Action Recommendations based on Predictive Analytics
def prescriptive_analytics(predicted_value, threshold=0.7):
    st.subheader("Prescriptive Analytics: Recommended Actions")
    if predicted_value > threshold:
        st.write("High risk of failure detected. Recommended Action: Schedule Maintenance.")
    else:
        st.write("Low risk of failure detected. Recommended Action: Continue Monitoring.")

# Sidebar to select analytics type
analytics_type = st.sidebar.radio(
    "Select the type of analytics to perform:",
    ('Descriptive', 'Diagnostic', 'Predictive', 'Prescriptive')
)

# Execute the selected analytics type
if analytics_type == 'Descriptive':
    descriptive_analytics(data)

elif analytics_type == 'Diagnostic':
    diagnostic_analytics(data)

elif analytics_type == 'Predictive':
    model = predictive_analytics(data, 'failure_risk')
    simulated_predicted_value = model.predict([data.drop('failure_risk', axis=1).iloc[0]])
    st.write(f"Simulated Predicted Failure Risk: {simulated_predicted_value[0]}")

elif analytics_type == 'Prescriptive':
    model = predictive_analytics(data, 'failure_risk')
    simulated_predicted_value = model.predict([data.drop('failure_risk', axis=1).iloc[0]])
    prescriptive_analytics(simulated_predicted_value[0])