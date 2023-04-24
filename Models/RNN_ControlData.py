#!/usr/bin/env python
# coding: utf-8

# <h2>Importing libraries</h2>

# In[1]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import RMSprop
from keras.optimizers import Adam


# Reading the CSV files that belong to the control group (EyeT_group_dataset_II_*.csv) from the dataset directory, appending them all and saving them as control_csv.csv. Here we make sure to check if a file named control_csv.csv, exists in the directory before creating a new file.

# In[2]:


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

# In[9]:


control_csv.head()


# Identifying all the unique "Participant name" values in the control_csv dataframe. Which returns the names of the paricipants in the control group.

# In[10]:


unique_participants_incontrol = control_csv["Participant name"].unique()
print(unique_participants_incontrol)


# Creating a function to input control_csv and test_csv dataframes as input and returns a numerical label based on the value in the 'Eye movement type' column. The function is converting categorical values (i.e., strings) to numeric labels which will be easier to use when training models.

# In[11]:


def event_label(row):
    labels = {'Saccade': 1, 'Fixation': 2, 'Unclassified': 3}
    return labels.get(row['Eye movement type'], 1)



# Implementing the function to select features from the control and test dataframes to create a new dataframe including them.

# In[12]:


def feature_selection(df):
    df = df.loc[:, ["Participant name", "Recording timestamp", "Eye movement type", "Gaze event duration"]]
        
    # Add column with numeric label for the eye movement type
    df.loc[df['Eye movement type'].isna(), 'Eye movement type'] = 'Unclassified'
    df.loc[df['Eye movement type'] == 'EyesNotFound', 'Eye movement type'] = 'Unclassified'
    df['Label'] = df.apply(lambda row: event_label(row), axis=1)

    
    return df[['Participant name', 'Recording timestamp', 'Label', 'Gaze event duration']]


# <h2>Add columns for empathy scores calculated before and after intervention.</h2>
# 
# Function implemented to add columns "empathy_before_std" (empathy scores before standard QCAE), "empathy_before_ext" (empathy scores before extended questionnaire), "empathy_after_std" (empathy scores after standard QCAE) and "empathy_after_ext" (empathy scores after extended questionnaire). 

# In[21]:


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


# <h1> Control data set </h1>
# 
# Invoking the feature_selection() function for the control_csv (control group)

# In[22]:


all_participants_control = feature_selection(control_csv)
print(all_participants_control)


# Invoking the add_empathy_scores() function for the control_csv (control group dataset) and saving the output as control_dataset

# In[23]:


control_dataset = add_empathy_scores(all_participants_control)


# <h3> Pre-processing </h3>

#  Reshaping the control_dataset array to a 2-dimensional array

# In[3]:


control_dataset = pd.read_csv('control_dataset_updated.csv')
control_dataset = np.reshape(control_dataset, (-1, 8))
control_dataset = pd.DataFrame(control_dataset, columns=['Participant name', 'Recording timestamp', 'Label', 'Gaze event duration', 'empathy_before_std', 'empathy_before_ext', 'empathy_after_std', 'empathy_after_ext'])
filename = 'control_dataset_updated.csv'
control_dataset.to_csv(filename, index=False)


# Check for the presence of NaN values in the control_dataset

# In[4]:


nan_counts = control_dataset.isna().sum()
print(nan_counts)


#  Filling the missing values in control_dataset using the fillna() method with the mean of each column

# In[5]:


control_dataset = control_dataset.fillna(df.mean())
nan_counts = control_dataset.isna().sum()
print(nan_counts)


# <h3> Timeseries plots </h3>
# 
# Plotting the "Recording timestamp" vs "eye movement type"

# In[6]:


# Filter the dataset for a time range
start_time = 4394847      
end_time = 4450000
filtered_dataset = control_dataset[(control_dataset['Recording timestamp'] >= start_time) & (control_dataset['Recording timestamp'] <= end_time)]

# Extract the 'Timestamp' and 'Eye movement type' columns 
timestamps = filtered_dataset['Recording timestamp']
event_labels = filtered_dataset['Label']

# Create a scatter plot of timestamp vs event label
plt.scatter(timestamps, event_labels)

# Label the x-axis and y-axis
plt.xlabel('timestamp')
plt.ylabel('Event Label')

# Display the plot
plt.show()


# Plotting the "Recording timestamp" vs "Gaze event duration"

# In[28]:


# Extract the 'Timestamp' and 'Gaze event duration' columns 
timestamps = control_dataset['Recording timestamp']
GazeEventDuration = control_dataset['Gaze event duration']

# Create a scatter plot of timestamp vs event label
plt.scatter(timestamps, GazeEventDuration)

# Label the x-axis and y-axis
plt.xlabel('Timestamp')
plt.ylabel('Gaze event duration')

# Display the plot
plt.show()


# <h2>LSTM-based RNN model  to predict the level of empathy of a participant</h2>

# In[7]:


# Split the data into training and testing sets
train_size = int(len(control_dataset) * 0.8)
train_data = control_dataset.iloc[:train_size]
test_data = control_dataset.iloc[train_size:]

print("Length of train_data:", len(train_data))
print("Length of test_data:", len(test_data))
# In[8]:


train_data.head()
# In[9]:


test_data.head()
# In[10]:


# Extract the gaze event duration and label columns from the dataset
X_train = np.array(train_data[['Gaze event duration', 'Label']])
y_train = np.array(train_data['empathy_after_std'])
X_test = np.array(test_data[['Gaze event duration', 'Label']])
y_test = np.array(test_data['empathy_after_std'])
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# In[11]:


# Normalize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[12]:


# Reshape the input data to fit the RNN model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[13]:


# Create the RNN model
model = Sequential()
model.add(LSTM(units=64, input_shape=(2, 1)))


# In[14]:


# Compile the RNN model
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae'])


# In[15]:


# Train the model
trained_results = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)


# In[16]:


model.summary()


# In[18]:


# Evaluate the model on the testing set
loss, mae = model.evaluate(X_test, y_test)

# Print the evaluation metrics
print("Test loss:", loss)
print("Test MAE:", mae)


# In[19]:
