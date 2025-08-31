import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------
# 1. DATA PREPROCESSING
# -----------------

print("Step 1: Starting data preprocessing...")

try:
    df = pd.read_csv('data/raw/plant_health_data.csv')
except FileNotFoundError:
    print("Error: The file 'plant_health_data.csv' was not found in the data/raw/ directory.")
    print("Please make sure the file is in the correct folder.")
    exit()

df_cleaned = df.dropna()

# We will handle the labels separately for the AI model
X_features = df_cleaned.drop(columns=['Timestamp', 'Plant_ID', 'Plant_Health_Status'])
y_labels = df_cleaned['Plant_Health_Status']

# Scale the numerical features (the 'X_features' dataframe)
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(X_features)
X_preprocessed = pd.DataFrame(scaled_features, columns=X_features.columns)

print("Data preprocessing complete!")

# -----------------
# 2. AI MODEL BUILDING
# -----------------

print("\nStep 2: Starting AI model building...")

# Split the data into a training set (80%) and a testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_labels, test_size=0.2, random_state=42)

# Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

print("AI model trained successfully!")

# -----------------
# 3. EVALUATE THE RESULTS
# -----------------

print("\nStep 3: Evaluating the model's performance...")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

print("\nDetailed Performance Report:")
print(classification_report(y_test, y_pred))

print("\nProject complete! You have successfully preprocessed the data and built an AI model.")
import joblib

# Save the trained model to a file
joblib.dump(model, 'myco_net_model.pkl')
print("\nAI model saved as 'myco_net_model.pkl'")
