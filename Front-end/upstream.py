import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV

# Streamlit app
st.title("Random Forest Regression with Feature Selection and Hyperparameter Tuning")

# Step 1: Load the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Step 2: Data Preprocessing
    # Handle missing values
    data = data.dropna()

    # Ask user for target column name
    target_column = st.text_input("Enter the target column name:", "MMP (mPa)")

    if target_column in data.columns:
        # Ensure target variable is numeric
        y = pd.to_numeric(data[target_column], errors='coerce').dropna()
        X = data.drop(columns=[target_column]).iloc[y.index]  # Align X with y indices

        # Handle any categorical variables in X
        X = pd.get_dummies(X, drop_first=True)

        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Feature selection using RFECV
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rfecv = RFECV(estimator=model_rf, step=1, cv=5, scoring='neg_mean_squared_error')
        rfecv.fit(X_train_scaled, y_train)

        selected_features = X.columns[rfecv.support_].tolist()
        st.write("Selected Features:")
        st.write(selected_features)

        # Reduce to selected features
        X_train_selected = rfecv.transform(X_train_scaled)
        X_test_selected = rfecv.transform(X_test_scaled)

        # Step 4: Hyperparameter tuning
        param_dist = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

        rf_random = RandomizedSearchCV(
            estimator=model_rf,
            param_distributions=param_dist,
            n_iter=50,
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=1,
            random_state=42
        )
        rf_random.fit(X_train_selected, y_train)

        st.write("Best Hyperparameters:")
        st.write(rf_random.best_params_)

        # Step 5: Evaluate model
        best_model = RandomForestRegressor(**rf_random.best_params_, random_state=42)
        best_model.fit(X_train_selected, y_train)
        y_pred = best_model.predict(X_test_selected)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"R-squared: {r2:.4f}")

        # Visualizations
        # Feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=selected_features, y=best_model.feature_importances_, palette='viridis')
        plt.title("Feature Importances")
        plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.clf()

        # Taylor Diagram
        plot_taylor_diagram(y_test, y_pred, "Taylor Diagram")
    else:
        st.error("Target column not found in the dataset. Please check the column name.")
else:
    st.info("Please upload a CSV file to proceed.")
