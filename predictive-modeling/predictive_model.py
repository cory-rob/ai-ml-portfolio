import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Sample business dataset
data = {
    "feature_1": [10, 15, 20, 25, 30, 35, 40, 45],
    "feature_2": [5, 7, 9, 11, 13, 15, 17, 19],
    "feature_3": [100, 120, 140, 160, 180, 200, 220, 240],
    "efficiency_score": [60, 65, 70, 74, 79, 83, 88, 92]
}

df = pd.DataFrame(data)

X = df[["feature_1", "feature_2", "feature_3"]]
y = df["efficiency_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict on new data
new_data = pd.DataFrame({
    "feature_1": [28],
    "feature_2": [12],
    "feature_3": [175]
})

future_prediction = model.predict(new_data)
print("Predicted efficiency score:", future_prediction[0])
