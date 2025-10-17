import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_processing import load_data, preprocess_data

# Load and preprocess data
data = load_data('data/sensor_data.csv')
data = preprocess_data(data)

# Split data
X = data.drop('failure', axis=1)
y = data['failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open('outputs/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save sample predictions
sample_pred = X_test.copy()
sample_pred['Predicted_Failure'] = y_pred
sample_pred.to_csv('outputs/prediction_sample.csv', index=False)
