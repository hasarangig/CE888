#!/usr/bin/env python
# coding: utf-8

# <h2>Importing libraries </h2>

# In[2]:


import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# Reading the CSV files that belong to the test group (EyeT_group_dataset_II*.csv) from the dataset directory, appending them all and saving them as test_csv.csv. Here we make sure to check if a file named tset_csv.csv, exists in the directory before creating a new file.

# <h2>Importing the dataset</h2>

# In[3]:


# Set the path to the directory containing the files
path = 'EyeT'

# Get a list of all the files starting with 'EyeT_group_dataset_II_' in the directory
file_list = [f for f in os.listdir(path) if f.startswith('EyeT_group_dataset_II_')]

# Create an empty list to store the dataframes
dfs = []
filename = 'control_csv.csv'

# Loop through each file and append the data to the list
for file in file_list:
    df = pd.read_csv(os.path.join(path, file))
    dfs.append(df)

# Concatenate the dataframes into a single dataframe
control_csv = pd.concat(dfs, ignore_index=True)

# Check if the file already exists
if os.path.exists(filename):
    print(f"The file {filename} already exists.")
else:
    # Save the concatenated dataframe to a file
    control_csv.to_csv(filename, index=False)
    print(f"The file {filename} has been saved.")


# Reading the the first 5 rows of a dataframe control_csv

# In[4]:


control_csv.head()


# <h2>Feature selection and conversion of categorical data to numerical data</h2>

# Creating a function to input control_csv and test_csv dataframes as input and returns a numerical label based on the value in the 'Eye movement type' column. The function is converting categorical values to numeric labels which will be easier to use when training models.

# In[5]:


def event_label(row):
    labels = {'Saccade': 1, 'Fixation': 2, 'Unclassified': 3}
    return labels.get(row['Eye movement type'], 1)


# Implementing the function to select features from the control and test dataframes to create a new dataframe including them.

# In[6]:


def feature_selection(df):
    df = df.loc[:, ["Participant name", "Recording timestamp", "Eye movement type", "Gaze event duration"]]
        
    # Add column with numeric label for the eye movement type
    df.loc[df['Eye movement type'].isna(), 'Eye movement type'] = 'Unclassified'
    df.loc[df['Eye movement type'] == 'EyesNotFound', 'Eye movement type'] = 'Unclassified'
    df['Label'] = df.apply(lambda row: event_label(row), axis=1)

    
    return df[['Participant name', 'Recording timestamp', 'Label', 'Gaze event duration']]


# <h2>Add new columns to the dataset</h2>
# 
# Function implemented to add columns "empathy_before_std" (empathy scores before standard QCAE), "empathy_before_ext" (empathy scores before extended questionnaire), "empathy_after_std" (empathy scores after standard QCAE) and "empathy_after_ext" (empathy scores after extended questionnaire). 

# In[7]:


def add_empathy_scores(all_participants):
    columns_A = ['Participant nr','Total Score original','Total Score extended']
    columns_B = ['Participant nr','Total Score original','Total Score extended']
    
    all_participants_with_scores = []

    empathy_before = pd.read_csv('Questionnaire_datasetIA.csv', encoding='cp1252', usecols=columns_A, dtype={'Participant nr' : str})
    empathy_before = empathy_before.dropna(how='all')
    
    
    empathy_after = pd.read_csv('Questionnaire_datasetIB.csv', encoding='cp1252',  usecols=columns_B, dtype={'Participant nr' : str})
    
    
    if isinstance(all_participants, pd.DataFrame):
        all_participants = [all_participants]
        
    for participant in all_participants:
        for index, row in participant.iterrows():
            participant_name = row['Participant name']
            participant_number = int(participant_name.replace("Participant00", "").replace("Participant000", ""))
            print("participant_name & participant_number -----------",participant_name, participant_number) 
            
        
        
            for i in range(len(empathy_before)):
                if empathy_before['Participant nr'][i] == str(participant_number):
                    row['empathy_before_std'] = empathy_before['Total Score original'][i]
                    row['empathy_before_ext'] = empathy_before['Total Score extended'][i]
                    break

            for i in range(len(empathy_after)):
                if empathy_after['Participant nr'][i] == str(participant_number):
                    row['empathy_after_std'] = empathy_after['Total Score original'][i]
                    row['empathy_after_ext'] = empathy_after['Total Score extended'][i]
                    break

            all_participants_with_scores.append(row)

    return all_participants_with_scores


# Invoking the add_empathy_scores() function for the control_csv (control group dataset) and saving the output as control_dataset

# In[8]:


all_participants_control = feature_selection(control_csv)
print(all_participants_control)


# Calculating the correlation coefficients between two parameters "Label" - Eye movement type and "Gaze event duration"

# In[10]:


# Calculate correlation between Gaze event duration and Eye movement type
corr = all_participants_control[['Label', 'Gaze event duration']].corr()
print(corr)


# Invoking the add_empathy_scores() function for the control group dataset and saving the output as control_dataset

# In[140]:


control_dataset = add_empathy_scores(all_participants_control)


# <h2>Pre-processing data</h2>

# Reshaping the control_dataset array to a 2-dimensional array

# In[141]:


control_dataset = np.reshape(control_dataset, (-1, 8))
control_dataset = pd.DataFrame(control_dataset, columns=['Participant name', 'Recording timestamp', 'Label', 'Gaze event duration', 'empathy_before_std', 'empathy_before_ext', 'empathy_after_std', 'empathy_after_ext'])


# Filling the missing values in control_dataset using the fillna() method with the mean of each column 

# In[142]:


control_dataset = control_dataset.fillna(df.mean())
nan_counts = control_dataset.isna().sum()
print(nan_counts)


# Calculating the mean of empathy_before_std column and then a new column called 'empathy_level_before_std' is created based on a condition: if the value in the 'empathy_before_std' column is greater than or equal to the mean value of the column, then the corresponding value in the 'empathy_level_before_std' column is set to 'High', otherwise it is set to 'Low'.

# In[143]:


# Calculate the mean value of empathy_before_std column
mean_empathy_before_std = control_dataset['empathy_before_std'].mean()
print(mean_empathy_before_std)

# Create a new column empathy_level_before_std based on mean value
control_dataset['empathy_level_before_std'] = np.where(control_dataset['empathy_before_std'] >= mean_empathy_before_std, 'High', 'Low')


# Calculating the mean of empathy_before_std column and then a new column called 'mean_empathy_after_std' is created based on a condition: if the value in the 'empathy_after_std' column is greater than or equal to the mean value of the column, then the corresponding value in the 'mean_empathy_after_std' column is set to 'High', otherwise it is set to 'Low'.

# In[144]:


# Calculate the mean value of empathy_after_std column
mean_empathy_after_std = control_dataset['empathy_after_std'].mean()
print(mean_empathy_after_std)

# Create a new column empathy level_after_std based on mean value
control_dataset['empathy_level_after_std'] = np.where(control_dataset['empathy_after_std'] >= mean_empathy_after_std, 'High', 'Low')


# Printing the columns names and values saved in control_dataset and saving the content to a file called 'control_dataset.csv' to further inspections (To check if the new columns are saved with correct values).

# In[15]:


print(control_dataset)
filename = 'control_dataset.csv'
control_dataset.to_csv(filename, index=False)


# In[16]:


# Calculate correlation between Gaze event duration and Eye movement type
correlation_matrix = control_dataset[['Label', 'Gaze event duration', 'empathy_before_std', 'empathy_before_ext', 'empathy_after_std', 'empathy_after_ext']].corr()
print(correlation_matrix)


# <h2>Splitting dataset for training, validation, and testing </h2>

# Control_dataset is divided into train, validation and test data sets and the lengths are printed to check the subset sizes

# In[146]:


# Split the data into training (40%), validation (40%), and testing (30%) sets
train_size = int(len(control_dataset) * 0.4)
val_size = int(len(control_dataset) * 0.4)
test_size = int(len(control_dataset) * 0.3)

train_data = control_dataset.iloc[:train_size]
val_data = control_dataset.iloc[train_size:train_size+val_size]
test_data = control_dataset.iloc[train_size+val_size:]

print("Length of train_data:", len(train_data))
print("Length of val_data:", len(val_data))
print("Length of test_data:", len(test_data))


# <h2>Random Forest Model</h2>

# Defining the Random Forest model and then training the model using the training data. After that the model is used to predict the empathy level of participants in the validation set. The accuracy of the model on the validation set is computed using the accuracy_score function. Finally, the model is evaluated on the testing data, and the accuracy calculated and printed

# In[147]:


X_train = train_data['Gaze event duration']
y_train = train_data['empathy_level_before_std']
X_val = val_data['Gaze event duration']
y_val = val_data['empathy_level_before_std'] 
X_test = test_data['Gaze event duration']
y_test = test_data['empathy_level_before_std']

# Create label encoder object
le = LabelEncoder()

# Fit encoder to target classes
le.fit(y_train)

# Transform target classes to numerical values
y_train_enc = le.transform(y_train)
y_val_enc = le.transform(y_val)

# Train random forest model
model = RandomForestClassifier(n_estimators=100, max_features="sqrt")
model.fit(X_train.values.reshape(-1, 1), y_train_enc)
    
# Make predictions and compute accuracy
y_pred_enc = model.predict(X_val.values.reshape(-1, 1))
y_pred = le.inverse_transform(y_pred_enc)
val_data['predicted_empathy_level'] = y_pred
acc = accuracy_score(y_val, y_pred)
print(f"Accuracy of the training data: {acc}")

# Make predictions on test data
y_pred_test_enc = model.predict(X_test.values.reshape(-1, 1))

# Convert predicted labels to original string labels
y_pred_test = le.inverse_transform(y_pred_test_enc)

# Evaluate model performance
test_data['predicted_empathy_level'] = y_pred_test
acc2 = accuracy_score(test_data['empathy_level_before_std'], y_pred_test)
print(f"Accuracy on test data: {acc2}")

