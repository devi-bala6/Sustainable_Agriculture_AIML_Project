import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed data
df = pd.read_csv('data/preprocessed/preprocessed_plant_health_data.csv')

print("Preprocessed data loaded successfully.")
print(df.head())

# Separate features (X) and label (y)
# We drop the identifiers as they are not needed for training
X = df.drop(columns=['Timestamp', 'Plant_ID', 'Plant_Health_Status'])
y = df['Plant_Health_Status']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData has been split into training and testing sets.")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Create the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

print("\nAI model has been trained successfully!")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
