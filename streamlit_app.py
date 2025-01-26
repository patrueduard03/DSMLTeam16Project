import streamlit as st
import pandas as pd
import numpy as np
import os
import random

# For reproducibility
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
    """Set seeds for Python, NumPy, and TensorFlow to improve reproducibility."""
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
    df_ro_population = pd.read_excel(
        os.path.join(base_path, 'Data_source.xlsx'),
        sheet_name='Romania Population'
    )
    df_buc_population = pd.read_excel(
        os.path.join(base_path, 'Data_source.xlsx'),
        sheet_name='Bucharest Population'
    )
    df_ro_avg_income = pd.read_excel(
        os.path.join(base_path, 'Data_source.xlsx'),
        sheet_name='Average Income Romania'
    )
    df_buc_avg_income = pd.read_excel(
        os.path.join(base_path, 'Data_source.xlsx'),
        sheet_name='Average Income Bucharest'
    )
    df_inflation_rate = pd.read_excel(
        os.path.join(base_path, 'Data_source.xlsx'),
        sheet_name='Inflation Rate'
    )

    # Convert columns to numeric where applicable
    for df in [
        df_ro_population, df_buc_population,
        df_ro_avg_income, df_buc_avg_income, df_inflation_rate
    ]:
        for col in df.columns:
            if col not in ["Year", "Month"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # -----------------------------
    # Merge on [Year, Month] for Romania
    # -----------------------------
    df_romania = pd.merge(
        df_ro_population,
        df_ro_avg_income,
        on=["Year", "Month"],
        how="inner"
    )
    df_romania = pd.merge(
        df_romania,
        df_inflation_rate,
        on=["Year", "Month"],
        how="inner"
    )

    # -----------------------------
    # Merge on [Year, Month] for Bucharest
    # -----------------------------
    df_bucharest = pd.merge(
        df_buc_population,
        df_buc_avg_income,
        on=["Year", "Month"],
        how="inner"
    )
    df_bucharest = pd.merge(
        df_bucharest,
        df_inflation_rate,
        on=["Year", "Month"],
        how="inner"
    )

    # -----------------------------
    # Create a proper Date column
    # -----------------------------
    df_romania["Date"] = pd.to_datetime(
        dict(year=df_romania["Year"], month=df_romania["Month"], day=1)
    )
    df_bucharest["Date"] = pd.to_datetime(
        dict(year=df_bucharest["Year"], month=df_bucharest["Month"], day=1)
    )

    # -----------------------------
    # Compute Standard of Living
    # -----------------------------
    df_romania["Standard_of_Living_Nominal"] = (
            df_romania["Salary"] / df_romania["Monthly Population"]
    )
    df_romania["Standard_of_Living_Real"] = (
            df_romania["Standard_of_Living_Nominal"] /
            (1 + df_romania["Inflation_Rate"])
    )

    df_bucharest["Standard_of_Living_Nominal"] = (
            df_bucharest["Salary"] / df_bucharest["Monthly Population"]
    )
    df_bucharest["Standard_of_Living_Real"] = (
            df_bucharest["Standard_of_Living_Nominal"] /
            (1 + df_bucharest["Inflation_Rate"])
    )

    return df_romania, df_bucharest


# -----------------------------------------------------------------------------
# 2. PREPARE DATA (Split Train/Test, Scale) and Store in Session State
# -----------------------------------------------------------------------------
def prepare_data():
    """
    Split data (monthly) into:
      - Train: 2015–2020
      - Test:  2023–2024
    Scale features and store in session_state.
    """
    if "data_prepared" not in st.session_state:
        df_romania, df_bucharest = load_data()

        # --------------------------------
        # Filter data by Date
        # --------------------------------
        # Train: Jan 2015 – Dec 2022
        train_data = df_romania[
            (df_romania["Date"] >= "2015-01-01") &
            (df_romania["Date"] <= "2022-12-01")
            ]
        # Test: Jan 2023 – Dec 2024
        test_data = df_romania[
            (df_romania["Date"] >= "2023-01-01") &
            (df_romania["Date"] <= "2024-12-01")
            ]

        # --------------------------------
        # Select features (X) and target (y)
        # --------------------------------
        X_train = train_data[["Monthly Population", "Inflation_Rate", "Salary"]]
        y_train = train_data["Standard_of_Living_Real"]

        X_test = test_data[["Monthly Population", "Inflation_Rate", "Salary"]]
        y_test = test_data["Standard_of_Living_Real"]

        # --------------------------------
        # Scale
        # --------------------------------
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --------------------------------
        # Store in session_state
        # --------------------------------
        st.session_state["data_prepared"] = True
        st.session_state["data_romania"] = df_romania
        st.session_state["data_bucharest"] = df_bucharest
        st.session_state["train_data"] = train_data
        st.session_state["test_data"] = test_data

        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test

        st.session_state["X_train_scaled"] = X_train_scaled
        st.session_state["X_test_scaled"] = X_test_scaled
        st.session_state["scaler"] = scaler


# -----------------------------------------------------------------------------
# 3. GET OR TRAIN LINEAR REGRESSION MODEL
# -----------------------------------------------------------------------------
def get_lr_model():
    """
    Return a trained Linear Regression model (cached in session_state).
    Also store predictions & metrics in session_state.

    Model Explanation:
    - Using scikit-learn's default LinearRegression()
    - No regularization or hyperparameter tuning.
    - It's a simple OLS (ordinary least squares) approach.
    """
    if "lr_model" not in st.session_state:
        lr_model = LinearRegression()
        lr_model.fit(st.session_state["X_train_scaled"], st.session_state["y_train"])
        st.session_state["lr_model"] = lr_model

        # Predict on test set
        lr_pred = lr_model.predict(st.session_state["X_test_scaled"])
        st.session_state["lr_pred"] = lr_pred

        # Compute metrics once
        lr_mae = mean_absolute_error(st.session_state["y_test"], lr_pred)
        lr_mse = mean_squared_error(st.session_state["y_test"], lr_pred)
        lr_r2 = r2_score(st.session_state["y_test"], lr_pred)

        st.session_state["lr_mae"] = lr_mae
        st.session_state["lr_mse"] = lr_mse
        st.session_state["lr_r2"] = lr_r2

    return st.session_state["lr_model"]


# -----------------------------------------------------------------------------
# 4. GET OR TRAIN NEURAL NETWORK MODEL
# -----------------------------------------------------------------------------
def get_nn_model():
    """
    Return a trained Neural Network model (cached in session_state).
    Store predictions & metrics in session_state.

    Model Explanation:
    - 3 hidden layers: 128 + 64 + 32 neurons (ReLU).
    - Output layer: single neuron, linear activation.
    - Custom Adam(learning_rate=0.001), MSE loss, MAE metric.
    - Up to 300 epochs, but uses EarlyStopping if val_loss doesn't improve.
    - Smaller batch_size can also improve training stability.
    """
    set_seeds(42)  # For reproducibility

    if "nn_model" not in st.session_state:
        from tensorflow.keras.callbacks import EarlyStopping

        nn_model = Sequential([
            Dense(128, activation='relu', input_shape=(st.session_state["X_train_scaled"].shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        # Custom Adam with a chosen learning rate (try 0.0005 or 0.001)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        nn_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,            # Stop if val_loss doesn't improve after 10 epochs
            restore_best_weights=True
        )

        # Train the model
        history = nn_model.fit(
            st.session_state["X_train_scaled"],
            st.session_state["y_train"],
            epochs=300,             # Higher max epochs
            batch_size=16,          # Decrease batch size to see if it helps
            validation_split=0.2,
            verbose=1,             # Show progress in terminal/log
            callbacks=[early_stop]
        )

        # Store the trained model
        st.session_state["nn_model"] = nn_model
        st.session_state["nn_history"] = history.history

        # Predict on test set
        nn_pred = nn_model.predict(st.session_state["X_test_scaled"]).flatten()
        st.session_state["nn_pred"] = nn_pred

        # Compute metrics once
        nn_mae = mean_absolute_error(st.session_state["y_test"], nn_pred)
        nn_mse = mean_squared_error(st.session_state["y_test"], nn_pred)
        test_loss, test_mae = nn_model.evaluate(
            st.session_state["X_test_scaled"],
            st.session_state["y_test"],
            verbose=0
        )

        st.session_state["nn_mae"] = nn_mae
        st.session_state["nn_mse"] = nn_mse
        st.session_state["nn_test_loss"] = test_loss
        st.session_state["nn_test_mae"] = test_mae

    return st.session_state["nn_model"]


# -----------------------------------------------------------------------------
# 5. STREAMLIT APP
# -----------------------------------------------------------------------------
def main():
    # Set seeds once at app start for reproducibility of the neural network
    set_seeds(42)

    # Prepare data and store it in session_state if needed
    prepare_data()

    # Sidebar
    st.sidebar.title("Options")
    option = st.sidebar.selectbox(
        "Choose an option:",
        ["Explore Data", "Linear Regression", "Neural Network", "Model Comparison"]
    )

    if st.sidebar.button("Reset the app"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()

    # Retrieve data from session_state
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

        st.write("### Romania Data (Monthly)")
        st.write(data_romania.head())

        st.write("### Bucharest Data (Monthly)")
        st.write(data_bucharest.head())

        # -------------------------------------------------------------------------
        # Plot Romania population over time
        # -------------------------------------------------------------------------
        st.write("#### Population Trend (Romania)")
        romania_pop = data_romania.set_index("Date")["Monthly Population"]
        st.line_chart(romania_pop)

        # -------------------------------------------------------------------------
        # Plot Bucharest population over time
        # -------------------------------------------------------------------------
        st.write("#### Population Trend (Bucharest)")
        bucharest_pop = data_bucharest.set_index("Date")["Monthly Population"]
        st.line_chart(bucharest_pop)

        # -------------------------------------------------------------------------
        # Plot Salary Trend (Romania)
        # -------------------------------------------------------------------------
        st.write("#### Salary Trend (Romania)")
        romania_salary = data_romania.set_index("Date")["Salary"]
        st.line_chart(romania_salary)

        # -------------------------------------------------------------------------
        # Plot Salary Trend (Bucharest)
        # -------------------------------------------------------------------------
        st.write("#### Salary Trend (Bucharest)")
        bucharest_salary = data_bucharest.set_index("Date")["Salary"]
        st.line_chart(bucharest_salary)

        # -------------------------------------------------------------------------
        # Plot Inflation Trend (Romania)
        # -------------------------------------------------------------------------
        st.write("#### Inflation Rate Trend (Romania)")
        romania_inflation = data_romania.set_index("Date")["Inflation_Rate"]
        st.line_chart(romania_inflation)

        # -------------------------------------------------------------------------
        # Plot Inflation Trend (Bucharest)
        # -------------------------------------------------------------------------
        st.write("#### Inflation Rate Trend (Bucharest)")
        bucharest_inflation = data_bucharest.set_index("Date")["Inflation_Rate"]
        st.line_chart(bucharest_inflation)

    elif option == "Linear Regression":
        st.title("Linear Regression")

        # Ensure LR model & predictions exist
        lr_model = get_lr_model()
        lr_pred = st.session_state["lr_pred"]

        # Retrieve metrics
        mae = st.session_state["lr_mae"]
        mse = st.session_state["lr_mse"]
        r2 = st.session_state["lr_r2"]

        st.write("**Model Explanation**:")
        st.write("- Simple **Linear Regression** model (no regularization).")
        st.write("- Features: Monthly Population, Inflation Rate, Salary.")
        st.write("- Target: Standard_of_Living_Real.")
        st.write(f"Training period: 2015-01 to 2020-12 ({len(train_data)} months)")
        st.write(f"Testing period: 2023-01 to 2024-12 ({len(test_data)} months)")

        # Show intercept and coefficients
        intercept = lr_model.intercept_
        coefficients = lr_model.coef_  # An array, one value per feature
        st.write(f"**Intercept**: {intercept:.6f}")
        st.write(f"**Coefficients** (Slope terms for each feature): {coefficients}")

        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # Plot actual vs. predicted
        fig, ax = plt.subplots()

        # Scatter plot: x = actual, y = predicted
        ax.scatter(y_test, lr_pred, color="blue", label="Predicted Points")

        # Plot perfect 1:1 line (dashed black line)
        min_val = min(y_test.min(), lr_pred.min())
        max_val = max(y_test.max(), lr_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2,
                label="Perfect Prediction (y = x)")

        ax.set_xlabel("Actual (Standard_of_Living_Real)")
        ax.set_ylabel("Predicted")
        ax.set_title("Linear Regression: Actual vs. Predicted")
        ax.legend()
        st.pyplot(fig)

    elif option == "Neural Network":
        st.title("Neural Network")

        # Ensure NN model & predictions exist
        nn_model = get_nn_model()
        nn_pred = st.session_state["nn_pred"]

        # Retrieve metrics
        nn_mae = st.session_state["nn_mae"]
        nn_mse = st.session_state["nn_mse"]
        nn_test_loss = st.session_state["nn_test_loss"]
        nn_test_mae = st.session_state["nn_test_mae"]

        st.write("**Model Explanation**:")
        st.write("- 2 hidden layers (64 + 32 neurons) with ReLU.")
        st.write("- 1 output layer (linear).")
        st.write("- **Adam** optimizer, **MSE** loss, **MAE** metric.")
        st.write("- Trained for 100 epochs, batch_size=32, 20% validation split.")
        st.write("- Seed set for reproducibility, but minor variations can still occur.")
        st.write(f"Training period: 2015-01 to 2020-12 ({len(train_data)} months)")
        st.write(f"Testing period: 2023-01 to 2024-12 ({len(test_data)} months)")

        st.write(f"**Test Loss (MSE) from Keras Evaluate:** {nn_test_loss:.4f}")
        st.write(f"**Test MAE from Keras Evaluate:** {nn_test_mae:.4f}")
        st.write(f"**Mean Absolute Error (MAE) from Sklearn:** {nn_mae:.4f}")
        st.write(f"**Mean Squared Error (MSE) from Sklearn:** {nn_mse:.4f}")

        # Display entire training history (100 epochs) from session_state
        history_data = st.session_state["nn_history"]
        fig, ax = plt.subplots()
        ax.plot(history_data['loss'], label='Train Loss')
        ax.plot(history_data['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title("Neural Network Training History (MSE Loss)")
        ax.legend()
        st.pyplot(fig)

    elif option == "Model Comparison":
        st.title("Model Comparison")

        # Ensure both LR and NN are loaded, along with their predictions
        lr_model = get_lr_model()
        lr_pred = st.session_state["lr_pred"]

        nn_model = get_nn_model()
        nn_pred = st.session_state["nn_pred"]

        # Plot actual vs predicted for both
        fig, ax = plt.subplots()
        ax.plot(test_data["Date"], y_test, label="Actual", marker='o')
        ax.plot(test_data["Date"], lr_pred, label="Linear Regression", linestyle='--')
        ax.plot(test_data["Date"], nn_pred, label="Neural Network", linestyle=':')
        ax.set_xlabel("Date")
        ax.set_ylabel("Standard of Living (Real)")
        ax.set_title("Model Comparison: Linear Regression vs Neural Network")
        ax.legend()
        st.pyplot(fig)

        # ------------------------------------------------
        # Compare metrics
        # ------------------------------------------------
        lr_mae = st.session_state["lr_mae"]
        lr_mse = st.session_state["lr_mse"]
        nn_mae = st.session_state["nn_mae"]
        nn_mse = st.session_state["nn_mse"]

        st.subheader("Error Metrics Comparison")
        st.write(f"- **Linear Regression**: MAE = {lr_mae:.4f}, MSE = {lr_mse:.6f}")
        st.write(f"- **Neural Network**: MAE = {nn_mae:.4f}, MSE = {nn_mse:.6f}")

        # Decide which model has lower error for each metric
        best_mae = "Linear Regression" if lr_mae < nn_mae else "Neural Network"
        best_mse = "Linear Regression" if lr_mse < nn_mse else "Neural Network"

        st.write(f"**MAE Winner:** {best_mae}")
        st.write(f"**MSE Winner:** {best_mse}")

        # ------------------------------------------------
        # Show table with actual vs. predicted
        # ------------------------------------------------
        st.write("### Actual vs. Predicted for Test Period (2023–2024)")
        test_results = test_data.copy()
        test_results["LR_Pred"] = lr_pred
        test_results["NN_Pred"] = nn_pred

        # Display only relevant columns
        display_cols = ["Date", "Standard_of_Living_Real", "LR_Pred", "NN_Pred"]
        st.dataframe(test_results[display_cols].reset_index(drop=True))

        # ------------------------------------------------
        # Additional line chart comparing columns side-by-side
        # ------------------------------------------------
        chart_df = test_results[["Date", "Standard_of_Living_Real", "LR_Pred", "NN_Pred"]].copy()
        chart_df = chart_df.set_index("Date")
        st.line_chart(chart_df)

if __name__ == "__main__":
    main()
