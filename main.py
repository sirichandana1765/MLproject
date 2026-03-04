import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pickle

# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("data/train.csv")

# Convert date column
df['date'] = pd.to_datetime(df['date'])

# =========================
# 2️⃣ Feature Engineering
# =========================

# Time Features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Sort for lag features (group only by store)
df = df.sort_values(['store', 'date'])

# Lag Features
df['lag_7'] = df.groupby('store')['sales'].shift(7)
df['lag_14'] = df.groupby('store')['sales'].shift(14)

# Rolling Mean
df['rolling_mean_7'] = df.groupby('store')['sales']\
                          .shift(1).rolling(7).mean()

# Drop missing rows from lag
df = df.dropna()

# =========================
# 3️⃣ Add Synthetic Price
# =========================
np.random.seed(42)
df['price'] = 10 + np.random.normal(0, 1, len(df))
df['cost'] = df['price'] * 0.6

# =========================
# 4️⃣ Model Training
# =========================

features = [
    'store',
    'price',
    'promo',
    'holiday',
    'year',
    'month',
    'day_of_week',
    'is_weekend',
    'lag_7',
    'lag_14',
    'rolling_mean_7'
]

X = df[features]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, predictions))

# Save model
with open("demand_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully ✅")