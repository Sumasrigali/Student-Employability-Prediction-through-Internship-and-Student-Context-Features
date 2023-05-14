#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


# Save the preprocessed dataset to a file
df.to_csv('preprocessed_Book1.csv')
df.head()


# In[4]:


X = df[['Internship%','CGPA','Specialized','LeadershipSkills','CommunicationSkills','TeamWork','Overall%']]
y = df['Result']


# In[7]:


from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
catboost_model = CatBoostClassifier(iterations=100, learning_rate=0.5)
catboost_model.fit(X_train, y_train, cat_features=[2, 3])
y_pred = catboost_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
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


# In[10]:


df.to_csv('preprocessed_Book1.csv')
catboost_model.load_model('catboost_model.cbm')
# Get input from user
internship_percent = int(input("Enter your internship percentage: "))
cgpa = float(input("Enter your CGPA: "))
specialization = input("Enter your specialization: ")
leadership_skills = int(input("Enter your leadership skills score (out of 10): "))
communication_skills = int(input("Enter your communication skills score (out of 10): "))
teamwork = int(input("Enter your teamwork score (out of 10): "))
# Calculate the overall percentage based on the input features
overall = (internship_percent + (cgpa*10) + (leadership_skills*10) + (communication_skills*10) + (teamwork*10)) / 5.0

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

# Make the prediction using the CatBoost classifier
prediction = catboost_model.predict(user_df)

if prediction[0] == 1:
    print("Congratulations! You are likely to be hired.")
else:
    print("Sorry, you are not likely to be hired.")


