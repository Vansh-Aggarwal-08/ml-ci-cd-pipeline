import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# load dataset
data = pd.read_csv('data\iris.csv')

# preprocess the dataset
X = data.drop('species', axis-1)
y = data ['species']

#Split the data into training and test sets 
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random state=42)  

# load the save model
model = joblib.load('model/iris_model.pkl')

# make prediction
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')

