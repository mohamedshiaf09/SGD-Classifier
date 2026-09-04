# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the **Iris dataset** and separate the features and target species.
2. Split the dataset into **training and testing sets**.
3. Train an **SGD Classifier** using the training data.
4. Predict the Iris species for the test data and **evaluate the accuracy**.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: MOHAMED SHIAF N
RegisterNumber: 212225240086
*/
```
```
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = SGDClassifier(max_iter=1000, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
<img width="935" height="245" alt="image" src="https://github.com/user-attachments/assets/5097bccc-bd4c-4af2-8c20-bf371bff6299" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
