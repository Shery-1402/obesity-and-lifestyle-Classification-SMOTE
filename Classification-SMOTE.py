#!/usr/bin/env python
# coding: utf-8

# # Import all the required libraries

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading dataset

# In[2]:


dataset=pd.read_csv('obesity_and_lifestyle dataset.csv')


# In[3]:


dataset.shape
# dataset.shape will return a tuple where the first element is the number of rows, and the second element is the number of columns


# # Explore the dataset

# In[4]:


dataset.head()
# in order to analyse the first 5 rows of the dataset. 
# This function is very useful for getting a quick overview of the data, especially when you are working with large datasets and you just want to see the first few rows to understand the structure of the DataFrame, the types of values it contains, and so on.


# In[5]:


dataset.info()


# In[6]:


dataset = dataset.rename(columns={'NObeyesdad': 'Obesity_level'}) #  renaming the target varible to a sensible name


# In[7]:


dataset['Obesity_level'].value_counts()


# In[8]:


# Set the figure size
plt.figure(figsize=(10, 6))

# Create the bar plot
ax = sns.countplot(data=dataset, x="Obesity_level", palette="Set2")

# labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.xlabel('Obesity Level', fontsize=12)
plt.ylabel('Count', fontsize=14)

# title
plt.title('Distribution of Obesity Levels', fontsize=16)

# Add annotations
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=10)


sns.despine()

plt.tight_layout()
plt.show()


# In[9]:


dataset.describe()


# #### Target variable

# In[10]:


dataset.info()


# #### Converting Categorical values to numerical

# In[11]:


dataset['Gender']=dataset['Gender'].replace(['Male', 'Female'],[1,0])
dataset['High_caloric_food']=dataset['High_caloric_food'].replace(['yes', 'no'],[1,0])
dataset['family_history_with_overweight']=dataset['family_history_with_overweight'].replace(['yes', 'no'],[1,0])
dataset['Smoke']=dataset['Smoke'].replace(['yes', 'no'],[1,0])
dataset['Consumption of food between meals']=dataset['Consumption of food between meals'].replace(['no', 'Sometimes','Frequently', 'Always'],[0,1,2,3])
dataset['Calories consumption monitoring']=dataset['Calories consumption monitoring'].replace(['yes', 'no'],[1,0])
dataset['Consumption of alcohol']=dataset['Consumption of alcohol'].replace(['no', 'Sometimes','Frequently', 'Always'],[0,1,2,3])
dataset['Transportation used']=dataset['Transportation used'].replace(['Walking','Public_Transportation', 'Automobile', 'Bike', 'Motorbike'],[0,1,2,3,4])
dataset['Obesity_level']=dataset['Obesity_level'].replace(['Normal_Weight','Obesity_Type_I','Insufficient_Weight','Overweight_Level_I','Overweight_Level_II','Obesity_Type_III','Obesity_Type_II'],[0,1,2,3,4,5,6])
dataset


# # Data exploratory analysis

# #### This is an essential investigation to perform on data to discover any patterns, spot anomalies and check assumptions with the help of summary statistics and graphical representations.

# In[12]:


sns.histplot(dataset.Age)
plt.title('Age distribution')
plt.show()


# In[13]:


plt.figure(figsize=(10,5))
plt.subplot(131)
sns.countplot(x= 'family_history_with_overweight', data=dataset, palette="GnBu_d",edgecolor="black")

plt.subplot(132)
sns.countplot(x= 'Smoke', data=dataset, palette="GnBu_d",edgecolor="black")


plt.subplot(133)
sns.countplot(x= 'High_caloric_food', data=dataset, palette="GnBu_d",edgecolor="black")


# In[14]:


plt.figure(figsize=(15,6))
plt.title ("Family History With Overweight")
sns.histplot(x="Age", hue="family_history_with_overweight", data=dataset)
plt.show()


# In[15]:


plt.figure(figsize=(15,6))
plt.title ("Transportation used'")
sns.histplot(x="Age", hue="Transportation used", data=dataset)
plt.show()


# In[16]:


plt.figure(figsize=(10,5))
plt.subplot(235)
sns.countplot(x= 'Transportation used', data=dataset, palette="GnBu_d",edgecolor="black")


# In[17]:


age_weight_sum = dataset.groupby('Age')['Weight'].sum()
plt.figure(figsize=(10,6))
age_weight_sum.plot(kind='bar')
plt.title('Total Weights by Age')
plt.xlabel('Age')
plt.ylabel('Total Weight')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# In[18]:


plt.figure(figsize=(15,6))
plt.title ("Family History With Overweight")
sns.histplot(x="Age", hue="Gender", data=dataset)
plt.show()


# In[19]:


# sns.scatterplot(x="Age", y="Weight", hue="Transportation used", data=dataset)
sns.histplot(x='Age', data=dataset, hue='Smoke', palette="GnBu_d",edgecolor="black")
plt.show()


# In[20]:


sns.histplot(x="Age", data=dataset, hue="Smoke", palette=["#FF0000", "#0000FF"], edgecolor="black")
plt.show()


# In[21]:


sns.histplot(x='Age', data=dataset, hue='Consumption of alcohol' ,palette="muted", edgecolor="black")
plt.show()


# In[22]:


sns.histplot(x='Age', y='Weight', data=dataset, hue='Frequency of consumption of vegetables',palette="muted", edgecolor="white")
plt.show()


# In[26]:


dataset.info()


# # import the dataset and assign the X and y

# In[25]:


dataset.shape


# In[27]:


data=dataset
X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]].values
y=data.iloc[:,16].values


# # split the data in train and test

# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
# len(X_train)


# In[37]:


from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# X variables = features and y = target variable, 70/30 split with random state at 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Oversampling the Minority Class using SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Training the model using KNN
classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(X_train_smote, y_train_smote)

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))


# # Evaluating the model
# 

# In[38]:


from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(cm,'\n\n')
print('-------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[39]:


ax = sns.heatmap(cm, cmap='flare', annot=True, fmt='d')
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("True Class", fontsize=12)
plt.title("Confusion Matrix", fontsize=12)
plt.show()

