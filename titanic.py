import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

# Checking the data
#print(df_train.dtypes)
#print(df_train.isna().sum())

# Group by Pclass and Sex and calculate the median age for each group
grouped = df_train.groupby(['Pclass', 'Sex'])['Age'].median()

# fill_age function to fill missing values of Age
def fill_age(row):
    if pd.isnull(row['Age']):
        return grouped[row['Pclass'], row['Sex']]
    else:
        return row['Age']

# Apply the function to fill missing ages
df_train['Age'] = df_train.apply(fill_age, axis=1)
df_test['Age'] = df_test.apply(fill_age, axis=1)

# Fill Embarked missing values with mode
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_test['Fare'].mode()[0], inplace=True)

# Extract the deck information from the cabin number
def get_deck(df):
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'].fillna('U', inplace=True)
    df.drop(columns=['PassengerId', 'Cabin', 'Name', 'Ticket'], inplace=True)
    
get_deck(df_train)
get_deck(df_test)

# Convert categorical variables
def convert_cat(df):
    categorical_var = ['Sex', 'Embarked', 'Deck']
    titanic = pd.get_dummies(df, columns=categorical_var)
    return titanic

titanic_train = convert_cat(df_train)
titanic_test = convert_cat(df_test)

# Drop Deck_T to align with testing set data
titanic_train.drop(columns=['Deck_T'], inplace=True)

df_train.to_csv('data/transformed_titanic.csv', index=False)
print()

# MODEL
# Separate the features 
X = titanic_train.drop('Survived', axis=1)
y = titanic_train['Survived']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define models and their hyperparameters
models = {
    'Random Forest': (RandomForestClassifier(random_state=42), {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10]
    }),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5]
    }),
    'SVM': (SVC(random_state=42), {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }),
    'KNN': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }),
    'Logistic Regression': (LogisticRegression(random_state=42), {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    })
}

# Perform Grid Search for each model
results = {}
best_models = {}

for name, (model, params) in models.items():
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    best_models[name] = grid_search.best_estimator_
    train_accuracy = accuracy_score(y_train, grid_search.predict(X_train_scaled))
    val_accuracy = accuracy_score(y_val, grid_search.predict(X_val_scaled))
    
    results[name] = {
        'Best Parameters': grid_search.best_params_,
        'Cross-validation Score': grid_search.best_score_,
        'Training Accuracy': train_accuracy,
        'Validation Accuracy': val_accuracy
    }
    print(f"{name}:")
    print(f"Best parameters: {results[name]['Best Parameters']}")
    print(f"Cross-validation Score: {results[name]['Cross-validation Score']:.4f}")
    print(f"Training Accuracy: {results[name]['Training Accuracy']:.4f}")
    print(f"Validation Accuracy: {results[name]['Validation Accuracy']:.4f}")
    #print(classification_report(y_val, grid_search.predict(X_val_scaled)))

# Create a table with scores
scores_df = pd.DataFrame({name: {'Training Accuracy': result['Training Accuracy'], 'Validation Accuracy': result['Validation Accuracy']} for name, result in results.items()}).T
scores_df = scores_df.sort_values('Validation Accuracy', ascending=False)
print("Model Performance Table:")
print(scores_df)

# Choose the best model
best_model_name = scores_df.index[0]
best_model = best_models[best_model_name]

print(f"\nBest Model: {best_model_name}")
print(f"Best Parameters: {results[best_model_name]['Best Parameters']}")

# Use the best model for final predictions
X_test_scaled = scaler.transform(titanic_test)
test_predictions = best_model.predict(X_test_scaled)

# Create submission DataFrame
id = range(892, 892 + len(test_predictions))
submission = pd.DataFrame({
    'PassengerId': id,
    'Survived': test_predictions
})

# Save CSV file
submission.to_csv('data/best_model_submission.csv', index=False)
