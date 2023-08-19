##Churn_modeling

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set_theme(color_codes=True)
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

dataset = pd.read_csv('Churn_Modelling.csv')

## Datapreprocessing 1
#cheak the number of unique value
print(dataset.select_dtypes(include="object").nunique())

#Feature Elimination
dataset.drop(columns = 'RowNumber',inplace = True)
dataset.drop(columns = 'CustomerId',inplace = True)
dataset.drop(columns = 'Surname',inplace = True)
dataset.head()

##Exploratory Data Analysis
#Store Catagorical Variable in one variable
Cat_var = ['Geography','Gender','Tenure','NumOfProducts', 'HasCrCard','IsActiveMember']

#Creating Fig with subplot
fig, axs = plt.subplots(nrows= 2, ncols= 3,figsize= (20,10))
axs= axs.flatten()

#Creating barplot for each catagorical variable
for i,var in enumerate(Cat_var):
    sns.barplot(x=var,y='Exited',data=dataset ,ax=axs[i])
    axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation = 90)
    
    #adjust spacing between subplots
    fig.tight_layout()

#show plot
plt.show()

# Create a grid of subplots based on the number of categorical variables
num_cat_vars = len(Cat_var)
num_cols = 2  # You can adjust the number of columns as needed
num_rows = (num_cat_vars + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

# Loop through categorical variables and create countplots
for i, var in enumerate(Cat_var):
    row = i // num_cols
    col = i % num_cols
    ax = axs[i]
    
    sns.countplot(data=dataset, x=var, hue='Exited', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_title(var)
    
# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# Create a grid of subplots based on the number of categorical variables
num_cat_vars = len(Cat_var)
num_cols = 2  # You can adjust the number of columns as needed
num_rows = (num_cat_vars + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

# Loop through categorical variables and create countplots
for i, var in enumerate(Cat_var):
    if i < len(axs):
        ax = axs[i]
        cat_counts = dataset[var].value_counts()
        ax.pie(cat_counts, labels=cat_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{var} Distribution')
    
# Adjust spacing between subplots
fig.tight_layout()

# If there's an extra subplot that needs to be removed
if len(axs) > num_cat_vars:
    fig.delaxes(axs[-1])

# Show the plot
plt.show()

num_vars = ['CreditScore','Age','Balance','EstimatedSalary']
# Create a grid of subplots based on the number of categorical variables
num_cat_vars = len(num_vars)
num_cols = 2  # You can adjust the number of columns as needed
num_rows = (num_cat_vars + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.boxplot(x=var,data=dataset, ax=axs[i])
    
# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# Create a grid of subplots based on the number of categorical variables
num_cat_vars = len(num_vars)
num_cols = 2  # You can adjust the number of columns as needed
num_rows = (num_cat_vars + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.violinplot(x=var,data=dataset, ax=axs[i])
    
# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

# Create a grid of subplots based on the number of categorical variables
num_cat_vars = len(num_vars)
num_cols = 2  # You can adjust the number of columns as needed
num_rows = (num_cat_vars + num_cols - 1) // num_cols

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
axs = axs.flatten()

for i, var in enumerate(num_vars):
    sns.histplot(x=var,data=dataset, ax=axs[i])
    
# Adjust spacing between subplots
fig.tight_layout()

# Show the plot
plt.show()

##Data Preprocessing 2
#Working with nulll values
print(dataset.isnull())
sns.heatmap(dataset.isnull() ,yticklabels=False , cmap="YlGnBu")
print(dataset.isnull().sum())
cheak_missing = dataset.isnull().sum() * 100/ dataset.shape[0]
print(cheak_missing[cheak_missing > 0].sort_values(ascending=False))

##Label Encoding:
for col in dataset.select_dtypes(include=['object']).columns:
    #print column name and value
    print(f"{col}:{dataset[col].unique()}")
    
from sklearn import preprocessing

#loop to find object datatype
for col in dataset.select_dtypes(include=['object']).columns:
    
    #initilization of LabelEncoder
    label_encoding= preprocessing.LabelEncoder()
    
    #fiting encoder to unique value
    label_encoding.fit(dataset[col].unique())
    
    #transform the column using encoder
    dataset[col] = label_encoding.transform(dataset[col])
    
    #print column name and new value
    print(f"{col}:{dataset[col].unique()}")
    
##Heatmap Correlation:
plt.figure(figsize=(15,12))
sns.heatmap(dataset.corr(),fmt='.2g',annot=True)
x = dataset.drop("Exited",axis=1)
y = dataset['Exited']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split( x, y, test_size=0.20, random_state=20)
from keras.models import Sequential
model=Sequential()
model.get_config()
from keras.layers import Dense
#layer 1 here mlops help you to hoe many input you take

model.add(Dense(
            units=6 ,
            input_dim = 10,
            kernel_initializer='zeros',
            bias_initializer='zeros',
    
        
) )

#layer 2
model.add(Dense(
            units=7 ,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            activation ='relu'
))

#layer 3
model.add(Dense(
            units=5 ,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            activation ='relu'
))

#layer 4
model.add(Dense(
            units=1 ,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            activation ='sigmoid'
))

model.compile(loss='binary_crossentropy')
model.get_weights()

model.fit(x,y,epochs=10)


# Predict probabilities on the test set
y_pred_probs = model.predict(x_test)

# Convert probabilities to class predictions
y_pred = (y_pred_probs > 0.5).astype(int)

 # Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision score
precision = precision_score(y_test, y_pred)

# Calculate recall score
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Print precision, recall, and F1 scores
print("Precision Score:", precision)
print("Recall Score:", recall)
print("F1 Score:", f1)

# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Print classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Calculate accuracy score (if not calculated before)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)
