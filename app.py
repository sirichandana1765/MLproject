import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# =========================
# Page Config
# =========================
st.set_page_config(page_title="Advanced Price Optimization Dashboard")
st.title("📊 Advanced ML Price Optimization System")
st.markdown("Interactive Demand Curve • Profit Optimization • Elasticity Analysis")

# =========================
# Load Model
# =========================
model = pickle.load(open("demand_model.pkl", "rb"))

# =========================
# User Inputs
# =========================

store = st.selectbox("Select Store ID", list(range(1, 11)))
price = st.slider("Select Price", 5.0, 20.0, 10.0)
promotion = st.selectbox("On Promotion?", [0, 1])
holiday = st.selectbox("Is Holiday?", [0, 1])
month = st.selectbox("Month", list(range(1, 13)))
day_of_week = st.selectbox("Day of Week (0=Mon)", list(range(0, 7)))

lag_7 = st.number_input("Last Week Sales", 0.0, 100000.0, 200.0)
lag_14 = st.number_input("2 Weeks Ago Sales", 0.0, 100000.0, 180.0)
rolling_mean_7 = st.number_input("7 Day Rolling Avg", 0.0, 100000.0, 190.0)

is_weekend = 1 if day_of_week >= 5 else 0
year = 2017

# =========================
# Base Input Data
# =========================

input_data = pd.DataFrame([{
    'store': store,
    'price': price,
    'promo': promotion,
    'holiday': holiday,
    'year': year,
    'month': month,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
    'lag_7': lag_7,
    'lag_14': lag_14,
    'rolling_mean_7': rolling_mean_7
}])

# =========================
# Generate Demand & Profit Curves
# =========================

prices = np.arange(5, 20, 0.5)
demands = []
profits = []

for p in prices:
    temp = input_data.copy()
    temp['price'] = p

    predicted_demand = model.predict(temp)[0]

    # Assume cost = 60% of price
    cost = p * 0.6
    predicted_profit = (p - cost) * predicted_demand

    demands.append(predicted_demand)
    profits.append(predicted_profit)

# =========================
# Find Optimal Price
# =========================

max_profit = max(profits)
best_price = prices[profits.index(max_profit)]
optimal_demand = demands[profits.index(max_profit)]

# =========================
# Demand Curve
# =========================

fig_demand = go.Figure()

fig_demand.add_trace(go.Scatter(
    x=prices,
    y=demands,
    mode='lines',
    name='Demand Curve'
))

fig_demand.add_trace(go.Scatter(
    x=[best_price],
    y=[optimal_demand],
    mode='markers',
    marker=dict(size=10),
    name='Optimal Point'
))

fig_demand.update_layout(
    title="📊 Demand Curve (Price vs Demand)",
    xaxis_title="Price",
    yaxis_title="Predicted Demand"
)

st.plotly_chart(fig_demand)

# =========================
# Profit Curve
# =========================

fig_profit = go.Figure()

fig_profit.add_trace(go.Scatter(
    x=prices,
    y=profits,
    mode='lines',
    name='Profit Curve'
))

fig_profit.add_trace(go.Scatter(
    x=[best_price],
    y=[max_profit],
    mode='markers',
    marker=dict(size=10),
    name='Max Profit Point'
))

fig_profit.update_layout(
    title="📈 Price vs Profit Curve",
    xaxis_title="Price",
    yaxis_title="Profit"
)

st.plotly_chart(fig_profit)

# =========================
# Elasticity Calculation
# =========================

base_demand = model.predict(input_data)[0]

delta_price = 0.5
temp = input_data.copy()
temp['price'] = price + delta_price

new_demand = model.predict(temp)[0]

percent_change_demand = (new_demand - base_demand) / base_demand
percent_change_price = delta_price / price

elasticity = percent_change_demand / percent_change_price

st.subheader("📈 Price Elasticity Analysis")
st.write(f"Elasticity Value: {round(elasticity, 3)}")

if abs(elasticity) > 1:
    st.warning("Demand is Elastic (Highly price sensitive)")
else:
    st.success("Demand is Inelastic (Less price sensitive)")

# =========================
# Final Results
# =========================

st.subheader("🔥 Optimization Result")
st.success(f"Optimal Price: ₹ {round(best_price,2)}")
st.info(f"Maximum Profit: ₹ {round(max_profit,2)}")