import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
df_train = pd.DataFrame(data_train)
df_test = pd.DataFrame(data_test)

# checking the data
print(df_train.dtypes)
print(df_train.isna().sum())

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

# fill Embarked missing values with mode
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_test['Fare'].fillna(df_test['Fare'].mode()[0],inplace=True)

# Extract the deck information from the cabin number
def get_deck(df):
    df['Deck'] = df['Cabin'].str[0]
    df['Deck'].fillna('U',inplace=True)
    df.drop(columns=['PassengerId','Cabin','Name','Ticket'],inplace=True)
    
get_deck(df_train)
get_deck(df_test)
   
#df_train.to_csv('data/cleaned_titanic.csv', index=False)

# Convert categorical variables
def convert_cat(df):
    categorical_var = ['Sex','Embarked','Deck']
    titanic = pd.get_dummies(df, columns=categorical_var)
    
    return titanic

titanic_train = convert_cat(df_train)
titanic_test = convert_cat(df_test)

# Drop Decl_T to align with testing set data
titanic_train.drop(columns=['Deck_T'],inplace=True)

df_train.to_csv('data/transformed_titanic.csv', index=False)
print()

# MODEL
# Seperate the features 
X = titanic_train.drop('Survived', axis=1)
y = titanic_train['Survived']

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.3, random_state=42)

# train the model
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)

# make predictions on validation set
val_predictions = rf_model.predict(X_val)

# Evaluate the model
print("Validation Accuracy:", accuracy_score(y_val, val_predictions))

id = [i for i in range(892, 892 + 418)]

# Test predictions 
test_predictions = rf_model.predict(titanic_test)

# Submission DataFrame
submission = pd.DataFrame({
    'PassengerId': id,
    'Survived': test_predictions    
})

# save csv file
submission.to_csv('data/gender_submission.csv', index=False)





