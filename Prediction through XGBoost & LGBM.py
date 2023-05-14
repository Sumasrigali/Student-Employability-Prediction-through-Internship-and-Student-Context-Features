#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('Book1.csv')

# Drop any rows with missing values
df.dropna(inplace=True)

# Convert the Days column to integer type
df['Days'] = df['Days'].astype(int)

# Remove any rows where the Internship column has a value less than 0 or greater than 100
df = df[(df['Internship%'] >= 0) & (df['Internship%'] <= 100)]

# Remove any rows where the CGPA column has a value less than 0 or greater than 10
df = df[(df['CGPA'] >= 0) & (df['CGPA'] <= 10)]

# Remove any rows where the LeadershipSkills, CommunicationSkills, or TeamWork columns have a value less than 0 or greater than 10
df = df[(df['LeadershipSkills'] >= 0) & (df['LeadershipSkills'] <= 10)]
df = df[(df['CommunicationSkills'] >= 0) & (df['CommunicationSkills'] <= 10)]
df = df[(df['TeamWork'] >= 0) & (df['TeamWork'] <= 10)]

# Remove any rows where the Overall% column has a value less than 0 or greater than 100
df = df[(df['Overall%'] >= 0) & (df['Overall%'] <= 100)]

# Convert the Result column to integer type
df['Result'] = df['Result'].astype(int)
df.head(35)


# In[4]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
df['Specialized'] = label_encoder.fit_transform(df['Specialized'])
# One-hot encode the 'Domain' column
one_hot_encoder = OneHotEncoder()
one_hot_encoded = one_hot_encoder.fit_transform(df[['Domain']])
df = df.join(pd.DataFrame(one_hot_encoded.toarray(), columns=one_hot_encoder.get_feature_names_out(['Domain'])))

# Drop the original 'Domain' column
df.drop('Domain', axis=1, inplace=True)

# Rename the one-hot encoded columns to remove the prefix 'Domain_'
df.rename(columns=lambda x: x.replace('Domain_', ''), inplace=True)

# Display the preprocessed dataset
print(df.head())


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

X = df[['Internship%','CGPA','Specialized','LeadershipSkills','CommunicationSkills','TeamWork','Overall%']]
y = df['Result']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model and fit it on the training set
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Use the model to make predictions on the testing set
y_pred = lr.predict(X_test)

# Calculate the accuracy score of the model
accuracy = accuracy_score(y_test, y_pred)
# Calculate precision score
precision = precision_score(y_test, y_pred)

# Calculate recall score
recall = recall_score(y_test, y_pred)

# Calculate f1-score
f1score = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)


# In[6]:


import joblib

# Load the trained model
lr = joblib.load('logistic_regression_model.joblib')

# Ask the user to input the data for the new student
internship_percentage = float(input("Enter the student's internship percentage: "))
cgpa = float(input("Enter the student's CGPA: "))
specialized = int(input("Enter the student's specialized code: "))
leadership_skills = float(input("Enter the student's leadership skills score: "))
communication_skills = float(input("Enter the student's communication skills score: "))
teamwork = float(input("Enter the student's teamwork score: "))

# Calculate the overall percentage based on the input features
overall = (internship_percentage + (cgpa*10) + (leadership_skills*10) + (communication_skills*10) + (teamwork*10)) / 5.0

df['Specialized'] = df['Specialized'].fillna('-1')
# Fit the encoder on the original specialization values
label_encoder.fit(df['Specialized'])
# Encode the user's specialization
specialization_encoded = label_encoder.transform([specialized])[0]


# Create a dataframe with the new student's data
new_student = pd.DataFrame({
    'Internship%': [internship_percentage],
    'CGPA': [cgpa],
    'Specialized': [specialized],
    'LeadershipSkills': [leadership_skills],
    'CommunicationSkills': [communication_skills],
    'TeamWork': [teamwork],
    'Overall%': [overall]
})

# Use the trained model to predict the employability of the new student
prediction = lr.predict(new_student)[0]
if prediction == 1:
    print("Congratulations! You are likely to be hired.")
else:
    print("Sorry, you are not likely to be hired.")


# In[7]:


from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CatBoost classifier and fit it on the training set
catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.5)
catboost_model.fit(X_train, y_train, cat_features=[2, 3])

# Make predictions on the testing set and calculate accuracy
y_pred = catboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Calculate recall score
recall = recall_score(y_test, y_pred)

# Calculate f1-score
f1score = f1_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1score)

score_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [accuracy, precision, recall, f1score]
colors = ['blue', 'red', 'green', 'purple']

plt.bar(score_names, scores, color=colors)
plt.title('Performance Scores')
plt.xlabel('Score Type')
plt.ylabel('Score')
plt.show()


# In[8]:


import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create XGBoost classifier and fit it on the training set
xgboost = xgb.XGBClassifier(objective='binary:logistic', max_depth=5, learning_rate=0.5)
xgboost.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy
y_pred = xgboost.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred)
precision_xgb = precision_score(y_test, y_pred)
recall_xgb = recall_score(y_test, y_pred)
f1score_xgb = f1_score(y_test, y_pred)

print("Accuracy:", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1-score:", f1score_xgb)

score_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [accuracy, precision, recall, f1score]
colors = ['blue', 'red', 'green', 'purple']

plt.bar(score_names, scores, color=colors)
plt.title('Performance Scores')
plt.xlabel('Score Type')
plt.ylabel('Score')
plt.show()


# In[9]:


import lightgbm as lgbm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LGBM classifier and fit it on the training set
lgbm_classifier = lgbm.LGBMClassifier(objective='binary', max_depth=5, learning_rate=0.2)
lgbm_classifier.fit(X_train, y_train)

# Make predictions on the testing set and calculate accuracy
y_pred = lgbm_classifier.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred)
precision_lgbm = precision_score(y_test, y_pred)
recall_lgbm = recall_score(y_test, y_pred)
f1score_lgbm = f1_score(y_test, y_pred)

print("Accuracy:", accuracy_lgbm)
print("Precision:", precision_lgbm)
print("Recall:", recall_lgbm)
print("F1-score:", f1score_lgbm)

score_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
scores = [accuracy, precision, recall, f1score]
colors = ['blue', 'red', 'green', 'purple']

plt.bar(score_names, scores, color=colors)
plt.title('Performance Scores')
plt.xlabel('Score Type')
plt.ylabel('Score')
plt.show()


# In[10]:


from tabulate import tabulate

# Create a list of dictionaries to store the metrics of the three classifiers
data = [
    {
        'Classifier': 'CatBoost',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1score
    },
    {
        'Classifier': 'XGBoost',
        'Accuracy': accuracy_xgb,
        'Precision': precision_xgb,
        'Recall': recall_xgb,
        'F1-score': f1score_xgb
    },
    {
        'Classifier': 'LGBM',
        'Accuracy': accuracy_lgbm,
        'Precision': precision_lgbm,
        'Recall': recall_lgbm,
        'F1-score': f1score_lgbm
    }
]

# Print the metrics in a tabular form
print(tabulate(data, headers='keys', tablefmt='fancy_grid'))


# In[11]:


import matplotlib.pyplot as plt
import numpy as np

# Set up data for plotting
score_names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
catboost_scores = [accuracy, precision, recall, f1score]
xgboost_scores = [accuracy_xgb, precision_xgb, recall_xgb, f1score_xgb]
lgbm_scores = [accuracy_lgbm, precision_lgbm, recall_lgbm, f1score_lgbm]

# Set up colors and bar widths
colors = ['blue', 'orange', 'green', 'purple']
width = 0.2

# Set up positions for the bars
positions_catboost = np.arange(len(score_names))
positions_xgboost = positions_catboost + width
positions_lgbm = positions_xgboost + width

# Create the figure and subplots
fig, ax = plt.subplots()

# Add the bars for each classifier
ax.bar(positions_catboost, catboost_scores, width, color=colors[0], label='CatBoost')
ax.bar(positions_xgboost, xgboost_scores, width, color=colors[1], label='XGBoost')
ax.bar(positions_lgbm, lgbm_scores, width, color=colors[2], label='LGBM')

# Set up the labels, title, and legend
ax.set_xticks(positions_xgboost)
ax.set_xticklabels(score_names)
ax.set_xlabel('Score Type')
ax.set_ylabel('Score')
ax.set_title('Performance Scores')
ax.legend()

# Show the plot
plt.show()


# In[13]:


# Get input from user
internship_percent = int(input("Enter your internship percentage: "))
cgpa = float(input("Enter your CGPA: "))
specialization = input("Enter your specialization: ")
leadership_skills = float(input("Enter your leadership skills score (out of 10): "))
communication_skills = float(input("Enter your communication skills score (out of 10): "))
teamwork = float(input("Enter your teamwork score (out of 10): "))
# Calculate the overall percentage based on the input features
overall = (internship_percentage + (cgpa*10) + (leadership_skills*10) + (communication_skills*10) + (teamwork*10)) / 5.0

# Set unknown values to -1
df['Specialized'] = df['Specialized'].fillna('-1')
# Fit the encoder on the original specialization values
label_encoder.fit(df['Specialized'])
# Encode the user's specialization
specialization_encoded = label_encoder.transform([specialization])[0]

# Create a dataframe with the user input
user_df = pd.DataFrame({
    'Internship%': [internship_percent],
    'CGPA': [cgpa],
    'Specialized': [specialization_encoded],
    'LeadershipSkills': [leadership_skills],
    'CommunicationSkills': [communication_skills],
    'TeamWork': [teamwork],
    'Overall%':[overall]
})

# Make the prediction using the XGBoost classifier
prediction = xgboost.predict(user_df)

if prediction[0] == 1:
    print("Congratulations! You are likely to be hired.")
else:
    print("Sorry, you are not likely to be hired.")


# In[17]:


# Get input from user
internship_percent = int(input("Enter your internship percentage: "))
cgpa = float(input("Enter your CGPA: "))
specialization = input("Enter your specialization: ")
leadership_skills = float(input("Enter your leadership skills score (out of 10): "))
communication_skills = float(input("Enter your communication skills score (out of 10): "))
teamwork = float(input("Enter your teamwork score (out of 10): "))
# Calculate the overall percentage based on the input features
overall = (internship_percentage + (cgpa*10) + (leadership_skills*10) + (communication_skills*10) + (teamwork*10)) / 5.0

# Set unknown values to -1
df['Specialized'] = df['Specialized'].fillna('-1')
# Fit the encoder on the original specialization values
label_encoder.fit(df['Specialized'])
# Encode the user's specialization
specialization_encoded = label_encoder.transform([specialization])[0]

# Create a dataframe with the user input
user_df = pd.DataFrame({
    'Internship%': [internship_percent],
    'CGPA': [cgpa],
    'Specialized': [specialization_encoded],
    'LeadershipSkills': [leadership_skills],
    'CommunicationSkills': [communication_skills],
    'TeamWork': [teamwork],
    'Overall%':[overall]
})

# Make the prediction using the XGBoost classifier
prediction = lgbm_classifier.predict(user_df)

if prediction[0] == 1:
    print("Congratulations! You are likely to be hired.")
else:
    print("Sorry, you are not likely to be hired.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




