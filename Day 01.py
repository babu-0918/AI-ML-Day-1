import pandas as pd

df = pd.read_csv('Titanic-Database.csv')

#Handling missing values

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Encoding categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'])

# Feature selection
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'] + \
           [col for col in df.columns if col.startswith('Sex_') or col.startswith('Embarked_')]
target = 'Survived'
X = df[features]
y = df[target]

# Splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Saving the model

import joblib
joblib.dump(model, 'titanic_model.pkl')

# Loading the model for future use

loaded_model = joblib.load('titanic_model.pkl')

# Example prediction with the loaded model

example_data = pd.DataFrame({
    'Pclass': [3],
    'Age': [22],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Sex_female': [0],
    'Sex_male': [1],
    'Embarked_C': [0],
    'Embarked_Q': [0],
    'Embarked_S': [1]
})
