import streamlit as st
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 0. SEED-SETTING FOR REPRODUCIBILITY
# -----------------------------------------------------------------------------
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# 1. DATA LOADING (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    base_path = os.path.join(os.getcwd(), 'data')

    # Read Excel sheets
    sheets = {
        "Romania Population": "df_ro_population",
        "Bucharest Population": "df_buc_population",
        "Average Income Romania": "df_ro_avg_income",
        "Average Income Bucharest": "df_buc_avg_income",
        "Inflation Rate": "df_inflation_rate"
    }

    dataframes = {}
    for sheet, var_name in sheets.items():
        dataframes[var_name] = pd.read_excel(os.path.join(base_path, 'Data_source.xlsx'), sheet_name=sheet)

    # Convert columns to numeric where applicable
    for df in dataframes.values():
        for col in df.columns:
            if col not in ["Year", "Month"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # Merge data for Romania and Bucharest
    df_romania = dataframes["df_ro_population"].merge(dataframes["df_ro_avg_income"], on=["Year", "Month"], how="inner")
    df_romania = df_romania.merge(dataframes["df_inflation_rate"], on=["Year", "Month"], how="inner")

    df_bucharest = dataframes["df_buc_population"].merge(dataframes["df_buc_avg_income"], on=["Year", "Month"], how="inner")
    df_bucharest = df_bucharest.merge(dataframes["df_inflation_rate"], on=["Year", "Month"], how="inner")

    # Create Date column
    for df in [df_romania, df_bucharest]:
        df["Date"] = pd.to_datetime(dict(year=df["Year"], month=df["Month"], day=1))

    # Compute Standard of Living
    for df in [df_romania, df_bucharest]:
        df["Standard_of_Living_Nominal"] = df["Salary"] / df["Monthly Population"]
        df["Standard_of_Living_Real"] = df["Standard_of_Living_Nominal"] / (1 + df["Inflation_Rate"])

    return df_romania, df_bucharest

# -----------------------------------------------------------------------------
# 2. PREPARE DATA
# -----------------------------------------------------------------------------
def prepare_data():
    if "data_prepared" not in st.session_state:
        df_romania, df_bucharest = load_data()

        train_data = df_romania[(df_romania["Date"] >= "2015-01-01") & (df_romania["Date"] <= "2022-12-01")]
        test_data = df_romania[(df_romania["Date"] >= "2023-01-01") & (df_romania["Date"] <= "2024-12-01")]

        features = ["Monthly Population", "Inflation_Rate", "Salary"]
        X_train, y_train = train_data[features], train_data["Standard_of_Living_Real"]
        X_test, y_test = test_data[features], test_data["Standard_of_Living_Real"]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.session_state.update({
            "data_prepared": True,
            "data_romania": df_romania,
            "train_data": train_data,
            "test_data": test_data,
            "X_train_scaled": X_train_scaled,
            "X_test_scaled": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "scaler": scaler
        })

# -----------------------------------------------------------------------------
# 3. LINEAR REGRESSION MODEL
# -----------------------------------------------------------------------------
def get_lr_model():
    if "lr_model" not in st.session_state:
        lr_model = LinearRegression()
        lr_model.fit(st.session_state["X_train_scaled"], st.session_state["y_train"])

        lr_pred = lr_model.predict(st.session_state["X_test_scaled"])

        st.session_state.update({
            "lr_model": lr_model,
            "lr_pred": lr_pred,
            "lr_mae": mean_absolute_error(st.session_state["y_test"], lr_pred),
            "lr_mse": mean_squared_error(st.session_state["y_test"], lr_pred),
            "lr_r2": r2_score(st.session_state["y_test"], lr_pred)
        })

    return st.session_state["lr_model"]

# -----------------------------------------------------------------------------
# 4. NEURAL NETWORK MODEL
# -----------------------------------------------------------------------------
def get_nn_model():
    set_seeds(42)

    if "nn_model" not in st.session_state:
        nn_model = Sequential([
            Dense(64, activation='relu', input_shape=(st.session_state["X_train_scaled"].shape[1],)),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        nn_model.fit(st.session_state["X_train_scaled"], st.session_state["y_train"], epochs=200, batch_size=32, validation_split=0.2, verbose=0)

        nn_pred = nn_model.predict(st.session_state["X_test_scaled"]).flatten()

        st.session_state.update({
            "nn_model": nn_model,
            "nn_pred": nn_pred,
            "nn_mae": mean_absolute_error(st.session_state["y_test"], nn_pred),
            "nn_mse": mean_squared_error(st.session_state["y_test"], nn_pred)
        })

    return st.session_state["nn_model"]

# -----------------------------------------------------------------------------
# 5. STREAMLIT APP (WITH COMPARISON)
# -----------------------------------------------------------------------------
def main():
    set_seeds(42)
    prepare_data()

    st.sidebar.title("Options")
    option = st.sidebar.radio("Select View:", ["Explore Data", "Linear Regression", "Neural Network", "Comparison"])

    if option == "Explore Data":
        st.write("### Data Overview")
        st.write(st.session_state["data_romania"])
        st.line_chart(st.session_state["data_romania"].set_index("Date")["Monthly Population"])

    elif option == "Linear Regression":
        get_lr_model()
        st.write(f"### Linear Regression Metrics\nMAE: {st.session_state['lr_mae']:.4f}\nMSE: {st.session_state['lr_mse']:.4f}\nR²: {st.session_state['lr_r2']:.4f}")

    elif option == "Neural Network":
        get_nn_model()
        st.write(f"### Neural Network Metrics\nMAE: {st.session_state['nn_mae']:.4f}\nMSE: {st.session_state['nn_mse']:.4f}")

    elif option == "Comparison":
        get_lr_model()
        get_nn_model()

        st.write("### Model Performance Comparison")
        st.write(f"📈 **Linear Regression**: MAE = {st.session_state['lr_mae']:.4f}, MSE = {st.session_state['lr_mse']:.4f}")
        st.write(f"🧠 **Neural Network**: MAE = {st.session_state['nn_mae']:.4f}, MSE = {st.session_state['nn_mse']:.4f}")

        fig, ax = plt.subplots()
        ax.plot(st.session_state["y_test"].values, label="Actual", linestyle="dashed")
        ax.plot(st.session_state["lr_pred"], label="Linear Regression")
        ax.plot(st.session_state["nn_pred"], label="Neural Network")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
