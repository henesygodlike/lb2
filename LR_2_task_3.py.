import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris_dataset = load_iris()
X = iris_dataset['data']
y = iris_dataset['target']
target_names = iris_dataset['target_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = SVC(kernel='rbf', gamma='auto')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print("Classification report:")
print(classification_report(y_test, predictions, target_names=target_names))

X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = model.predict(X_new)
print("Прогнозований клас:", target_names[prediction[0]])
