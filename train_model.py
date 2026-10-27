# train_model.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create Random Forest model
clf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
clf.fit(X_train, y_train)

# Save trained model
joblib.dump(clf, "iris_model.pkl")

print("Model saved as iris_model.pkl")