import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

# Checking the data
print(df_train.dtypes)
print(df_train.isna().sum())

# Group by Pclass and Sex and calculate the median age for each group
grouped = df_train.groupby(['Pclass', 'Sex'])['Age'].median()

# Fill missing Age values
def fill_age(row):
    if pd.isnull(row['Age']):
        return grouped[row['Pclass'], row['Sex']]
    else:
        return row['Age']

df_train['Age'] = df_train.apply(fill_age, axis=1)
df_test['Age'] = df_test.apply(fill_age, axis=1)

# Fill missing Embarked and Fare values
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
if 'Deck_T' in titanic_train.columns:
    titanic_train.drop(columns=['Deck_T'], inplace=True)

# Separate the features and target
X = titanic_train.drop('Survived', axis=1)
y = titanic_train['Survived']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
}

# Dictionary to store the accuracy results
results = {}

# Evaluate each model
for model_name, model in models.items():
    model.fit(X_train, y_train)
    val_predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, val_predictions)
    results[model_name] = accuracy

# Create a DataFrame with the results
results_df = pd.DataFrame(list(results.items()), columns=['Model Name', 'Accuracy'])

# Print the results
print(results_df)

# Save the results to a CSV file
results_df.to_csv('data/model_accuracies.csv', index=False)

# Select the best model based on accuracy
best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model Name']
best_model = models[best_model_name]

# Train the best model on the entire training set
best_model.fit(X, y)

# Make predictions on the test set
test_predictions = best_model.predict(titanic_test)

# Create a submission DataFrame
submission = pd.DataFrame({
    'PassengerId': [i for i in range(892, 892 + 418)],
    'Survived': test_predictions
})

# Save the submission file
submission.to_csv('data/gender_submission.csv', index=False)
