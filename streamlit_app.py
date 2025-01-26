import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
import os


# -------------------------
# 1. DATA LOADING
# -------------------------
@st.cache_data
def load_data():
    base_path = os.path.join(os.getcwd(), 'data')
    df_ro_population = pd.read_excel(os.path.join(base_path, 'Data_source.xlsx'), sheet_name='Romania Population')
    df_buc_population = pd.read_excel(os.path.join(base_path, 'Data_source.xlsx'), sheet_name='Bucharest Population')
    df_ro_avg_income = pd.read_excel(os.path.join(base_path, 'Data_source.xlsx'), sheet_name='Average Income Romania')
    df_buc_avg_income = pd.read_excel(os.path.join(base_path, 'Data_source.xlsx'),
                                      sheet_name='Average Income Bucharest')
    df_inflation_rate_avg_income = pd.read_excel(os.path.join(base_path, 'Data_source.xlsx'),
                                                 sheet_name='Inflation Rate')

    # Convert columns to numeric where applicable
    for df in [df_ro_population, df_buc_population, df_ro_avg_income, df_buc_avg_income]:
        for column in df.columns:
            if column != "Year":
                df[column] = pd.to_numeric(df[column], errors='coerce')

    # Merge datasets for Romania
    df_romania = pd.merge(df_ro_population, df_ro_avg_income, on="Year", how="inner")
    df_romania = pd.merge(df_romania, df_inflation_rate_avg_income, on="Year", how="inner")

    # Merge datasets for Bucharest
    df_bucharest = pd.merge(df_buc_population, df_buc_avg_income, on="Year", how="inner")
    df_bucharest = pd.merge(df_bucharest, df_inflation_rate_avg_income, on="Year", how="inner")

    # Calculate Standard of Living (Nominal)
    df_romania["Standard_of_Living_Nominal"] = df_romania["Salary"] / df_romania["Monthly Population"]
    df_bucharest["Standard_of_Living_Nominal"] = df_bucharest["Salary"] / df_bucharest["Monthly Population"]

    # Adjust Standard of Living for Inflation
    df_romania["Standard_of_Living_Real"] = df_romania["Standard_of_Living_Nominal"] / (
                1 + df_romania["Inflation_Rate"])
    df_bucharest["Standard_of_Living_Real"] = df_bucharest["Standard_of_Living_Nominal"] / (
                1 + df_bucharest["Inflation_Rate"])

    return df_romania, df_bucharest


# -------------------------
# 2. INITIALIZE SESSION STATE
# -------------------------
# This will ensure we only split and scale data once per session.
def prepare_data():
    """Split data into train/test and scale. Store in session_state."""
    if "data_prepared" not in st.session_state:
        data_romania, data_bucharest = load_data()

        # Filter data for training and testing
        train_data = data_romania[(data_romania["Year"] >= 2015) & (data_romania["Year"] <= 2022)]
        test_data = data_romania[(data_romania["Year"] > 2022)]

        # Prepare features and labels
        X_train = train_data[["Monthly Population", "Inflation_Rate", "Salary"]]
        y_train = train_data["Standard_of_Living_Real"]
        X_test = test_data[["Monthly Population", "Inflation_Rate", "Salary"]]
        y_test = test_data["Standard_of_Living_Real"]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Store in session state
        st.session_state["data_prepared"] = True
        st.session_state["data_romania"] = data_romania
        st.session_state["data_bucharest"] = data_bucharest
        st.session_state["train_data"] = train_data
        st.session_state["test_data"] = test_data
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["X_train_scaled"] = X_train_scaled
        st.session_state["X_test_scaled"] = X_test_scaled
        st.session_state["scaler"] = scaler


# -------------------------
# 3. GET OR TRAIN LINEAR REGRESSION MODEL
# -------------------------
def get_lr_model():
    """Return a trained Linear Regression model (cached in session_state)."""
    if "lr_model" not in st.session_state:
        # Train the model
        lr_model = LinearRegression()
        lr_model.fit(st.session_state["X_train_scaled"], st.session_state["y_train"])
        st.session_state["lr_model"] = lr_model
    return st.session_state["lr_model"]


# -------------------------
# 4. GET OR TRAIN NEURAL NETWORK MODEL
# -------------------------
def get_nn_model():
    """Return a trained Neural Network model (cached in session_state)."""
    if "nn_model" not in st.session_state:
        # Build the neural network
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(st.session_state["X_train_scaled"].shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train the model
        nn_model.fit(
            st.session_state["X_train_scaled"],
            st.session_state["y_train"],
            epochs=100,  # or your preferred number of epochs
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        st.session_state["nn_model"] = nn_model
    return st.session_state["nn_model"]


# -------------------------
# 5. STREAMLIT APP
# -------------------------
def main():
    # Prepare data and store it in session_state if needed
    prepare_data()

    # Sidebar
    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an option:",
                                  ["Explore Data", "Linear Regression", "Neural Network", "Model Comparison"])

    data_romania = st.session_state["data_romania"]
    data_bucharest = st.session_state["data_bucharest"]
    train_data = st.session_state["train_data"]
    test_data = st.session_state["test_data"]

    X_train_scaled = st.session_state["X_train_scaled"]
    y_train = st.session_state["y_train"]
    X_test_scaled = st.session_state["X_test_scaled"]
    y_test = st.session_state["y_test"]

    if option == "Explore Data":
        st.title("Explore Data")
        st.write("Dataset Preview (Romania):")
        st.write(data_romania)

        st.write("Dataset Preview (Bucharest):")
        st.write(data_bucharest)

        # Plot Romania population
        st.write("Population Trend (Romania):")
        st.line_chart(data_romania[["Year", "Monthly Population"]].set_index("Year"))

        # Plot Bucharest population
        st.write("Population Trend (Bucharest):")
        st.line_chart(data_bucharest[["Year", "Monthly Population"]].set_index("Year"))

    elif option == "Linear Regression":
        st.title("Linear Regression")

        # Get or train the linear regression model
        lr_model = get_lr_model()

        # Predictions
        y_pred = lr_model.predict(X_test_scaled)

        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Mean Absolute Error (MAE): {mae}")
        st.write(f"Mean Squared Error (MSE): {mse}")
        st.write(f"R² Score: {r2}")

        # Plot actual vs predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        st.pyplot(fig)

    elif option == "Neural Network":
        st.title("Neural Network")

        # Train or get from session_state
        nn_model = get_nn_model()

        # Evaluate on test set
        test_loss, test_mae = nn_model.evaluate(X_test_scaled, y_test, verbose=0)
        st.write(f"Test Loss (MSE): {test_loss}")
        st.write(f"Test MAE: {test_mae}")

        # If you want to show training curves, you can’t retrieve them after the session ends
        # unless you store the History object. For simplicity, let's do a quick re-train with fewer epochs
        # just for demonstration. But typically you'd keep the logs from the original training.

        # A simple approach: we'll re-train for a small number of epochs to illustrate the chart:
        # (In a real app, store the training history in session_state as well.)
        nn_model_temp = Sequential([
            Dense(64, activation='relu', input_shape=(st.session_state["X_train_scaled"].shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        nn_model_temp.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = nn_model_temp.fit(
            st.session_state["X_train_scaled"],
            st.session_state["y_train"],
            epochs=2,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        # Plot training history (for demonstration)
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Train Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

    elif option == "Model Comparison":
        st.title("Model Comparison")

        # Get both models
        lr_model = get_lr_model()
        nn_model = get_nn_model()

        # Predictions
        lr_pred = lr_model.predict(X_test_scaled)
        nn_pred = nn_model.predict(X_test_scaled).flatten()  # Flatten from (n,1) to (n,)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(test_data["Year"], y_test, label="Actual", marker='o')
        ax.plot(test_data["Year"], lr_pred, label="Linear Regression", linestyle='--')
        ax.plot(test_data["Year"], nn_pred, label="Neural Network", linestyle=':')
        ax.set_xlabel("Year")
        ax.set_ylabel("Standard of Living (Real)")
        ax.legend()
        st.pyplot(fig)

        # Metrics comparison
        lr_mae = mean_absolute_error(y_test, lr_pred)
        nn_mae = mean_absolute_error(y_test, nn_pred)

        st.write(f"Linear Regression MAE: {lr_mae}")
        st.write(f"Neural Network MAE: {nn_mae}")


if __name__ == "__main__":
    main()
