# Install scikit-learn if not already installed:
# !pip install scikit-learn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
# Preprocess (scale) the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Create and train a simple deep neural network
model = MLPClassifier(hidden_layer_sizes=(10, 8), activation='relu', max_iter=500, random_state=1)
model.fit(X_train, y_train)
# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predict a sample

sample = X_test[0].reshape(1, -1)

predicted_class = model.predict(sample)[0]

print("Predicted class:", iris.target_names[predicted_class])
