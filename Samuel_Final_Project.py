#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import the necessary libaries needed for the project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import statsmodels.api as sm
import scipy.stats as sci
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.linear_model import Ridge
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#import the csv file with pandas and the file path
df = pd.read_csv(r"C:\Users\Samuel\Desktop\project4\Hepatitis\Hepatitis.csv")
df.head(10)


# # please note: Die = 1 , Live =2 only in Class

# # Male = 1 , Female = 2 only in sex

# # 'no' = 1  and 'yes' =2 in any other variable

# #  ###CLEANING THE DATASET

# In[4]:


df.shape


# In[5]:


# i replaced the missing values with NaN to make things easier for when i want to drop
# NaN stands for Not a Number 
df.replace('?',np.nan,inplace = True)


# In[6]:


df.head()


# In[7]:


# Assign 'df.isnull' to 'miss_data'
miss_data = df.isnull()
miss_data.head()


# In[8]:


#Counting the amount of times a value appears in each column
# True stands for missing values
# False stands for the opposite
for x in miss_data.columns.values.tolist():
    print(x)
    print(miss_data[x].value_counts())
    print('')


# In[9]:


df.columns


# In[10]:


# droping an entire row if the value for steroid on that row is NaN 
df = df.dropna(subset = ['Steroid'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[11]:


# droping an entire row if the value for Fatigue on that row is NaN 
df = df.dropna(subset = ['Fatigue'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[12]:


# droping an entire row if the value for Malaise on that row is NaN 
df = df.dropna(subset = ['Malaise'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[13]:


# droping an entire row if the value for Anorexia on that row is NaN 
df = df.dropna(subset = ['Anorexia'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[14]:


# droping an entire row if the value for Liver Big on that row is NaN 
df = df.dropna(subset = ['Liver Big'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[15]:


# droping an entire row if the value for Liver Firm on that row is NaN 
df = df.dropna(subset = ['Liver Firm'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[16]:


# droping an entire row if the value for Spleen Palpable on that row is NaN 
df = df.dropna(subset = ['Spleen Palpable'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[17]:


# droping an entire row if the value for Spiders on that row is NaN 
df = df.dropna(subset = ['Spiders'],axis = 0)
df.reset_index(drop = True, inplace= True)


# In[18]:


df.shape


# In[19]:


df.head()


# ### Find the Average of which ever column has NaN and replace NaN with the average

# In[20]:


#finding the average of Bilirubin
avg_Bilirubin = df['Bilirubin'].astype('float').mean()
print('average of Bilirubin:', avg_Bilirubin)


# In[21]:


#Replacing NaN with the Average
df['Bilirubin'].replace(np.nan,avg_Bilirubin,inplace=True)


# ..

# In[22]:


avg_Alk_Phosphate = df['Alk Phosphate'].astype('float').mean()
print('average of Alk Phosphate :', avg_Alk_Phosphate)


# In[23]:


df['Alk Phosphate'].replace(np.nan,avg_Alk_Phosphate,inplace=True)


# ..

# In[24]:


#finding the average of Space
avg_SerumGt = df['SerumGt'].astype('float').mean()
print('average of :', avg_SerumGt)


# In[25]:


#Replacing NaN with the Average
df['SerumGt'].replace(np.nan,avg_SerumGt,inplace=True)


# ..

# In[26]:


avg_Albumin = df['Albumin'].astype('float').mean()
print('average of Albumin:', avg_Albumin)


# In[27]:


df['Albumin'].replace(np.nan,avg_Albumin,inplace=True)


# ..

# In[28]:


avg_Protime = df['Protime'].astype('float').mean()
print('average of Protime:', avg_Protime)


# In[29]:


df['Protime'].replace(np.nan,avg_Protime,inplace=True)


# In[30]:


check2_data = df.isnull()
for x in check2_data.columns.values.tolist():
    print(x)
    print(check2_data[x].value_counts())
    print('')


# # Visualization and analysis

# In[31]:


sns.pairplot(df)


# In[45]:


sns.scatterplot(x='Age', y='Protime', hue='Class', data=df, palette={1: 'red', 2: 'blue'}, legend='full')


# unique_classes = df['Class'].unique()
# 
# #Create a scatter plot for each unique class
# for class_value in unique_classes:
#     subset = df[df['Class'] == class_value]
#     plt.scatter(subset['Protime'], subset['Age'], label=class_value)
# 
# plt.xlabel('Protime')
# plt.ylabel('Age')
# plt.legend()
# plt.title("The relationship between Protime, Age and Class")
# plt.show()

# In[ ]:





# In[47]:


X = df[['Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise',
       'Anorexia', 'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders',
       'Ascites ', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SerumGt',
       'Albumin', 'Protime', 'Histology']].values
X[0:5]


# In[48]:


Y = df['Class'].values
Y[0:5]


# # Spliting data into training and testing data sets

# In[49]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X , Y, test_size=0.2 , random_state = 4)
print('Train Data ',X_train.shape,Y_train.shape)
print('Test Data',X_test.shape,Y_test.shape)


# # Classification of the model

# In[63]:


from sklearn.tree import DecisionTreeClassifier
hepTree = DecisionTreeClassifier(criterion="entropy",max_depth = 13)
hepTree


#     #### Training the model

# In[64]:


hepTree.fit(X_train,Y_train)


#  #### Predicting with the test data

# In[69]:


PredTree = hepTree.predict(X_test)
print(PredTree[0:5])
print(Y_test[0:5])


# # Accuracy evaluation

# In[71]:


from sklearn import metrics
print ('Train set accuracy :',metrics.accuracy_score(Y_train,hepTree.predict(X_train)) )
print ('Test set accuracy :',metrics.accuracy_score(Y_test,PredTree)) 


# In[72]:


print('The model is ',round(metrics.accuracy_score(Y_test,PredTree)*100),'% accurate') 


# ##KNN Model
# mean_acc = np.zeros((9))
# 
# for n in range(1,10):
#     neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train_norm,Y_train)
#     Yhat = neigh.predict(X_test_norm)
#     mean_acc[n-1] = metrics.accuracy_score(Y_test,Yhat) 
# 
# print(mean_acc)
# print('The best accuracy was with',mean_acc.max(),'with k =',mean_acc.argmax()+1)

# In[ ]:





# ### Finding out what has the most impact on the decision tree

# In[ ]:





# In[75]:


# Get feature importance
feature_importance = hepTree.feature_importances_

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance, align='center')
plt.yticks(range(len(feature_importance)),[ 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise',
       'Anorexia', 'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders',
       'Ascites ', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SerumGt',
       'Albumin', 'Protime', 'Histology'])  # Assuming you have a list of feature names
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Decision Tree Feature Importance')
plt.show()


# Albumin has the highest impact on the decision tree

# In[ ]:





# In[ ]:





# In[ ]:





# In[61]:


df.columns


# In[62]:


df.shape


# In[ ]:


while True:
    
    Age = (input('Please enter your Age:'))
    Sex = (input('Please enter your genderPlease note "1"= "Male" and "2" = "Female":'))
    Steroid = (input('Are you on the steroids treatment. Note:"1" = "no" and "2" = "yes":'))
    Antivirals = (input('Are you on Antiviral medication. Note:"1" = "no" and "2" = "yes" :')) 
    Fatigue = (input('Do you feel Fatigue alot. Note:"1" = "no" and "2" = "yes" :'))
    Malaise = (input('Do you feel Malaise. Note:"1" = "no" and "2" = "yes" :'))
    Anorexia = (input('Do you experience Anorexia. Note:"1" = "no" and "2" = "yes" :'))
    Liver_Big = (input('Do you have an enlarged liver. Note:"1" = "no" and "2" = "yes" :'))
    Liver_Firm = (input('Do you have a stiff liver. Note:"1" = "no" and "2" = "yes" :'))
    Spleen_Palpable = (input('Do you have an enlarged spleen. Note:"1" = "no" and "2" = "yes" :'))
    Spiders = (input('Are you experiencing Spider angiomas . Note:"1" = "no" and "2" = "yes" :'))
    Ascites = (input('Do you have accumulated fluid in your abdominal cavity. Note:"1" = "no" and "2" = "yes" :'))
    Varices = (input('have you been confirmed to have enlarged and swollen blood vessels in the esophagus. Note:"1" = "no" and "2" = "yes" :'))
    Bilirubin = (input('Please enter your Bilirubin measurement:'))
    Alk_Phosphate = (input('Please enter your Alk Phosphate measurement:'))
    SerumGt = (input('Please enter your SGOT measurement:'))
    Albumin = (input('Please enter your Albumin measurement:'))
    Protime = (input('Please enter your Prothrombin measurement:'))
    Histology = (input('Did your Histology report come out as positive. Note:"1" = "no" and "2" = "yes" :'))
                                                                        
        
    X_data = np.array([[Age, Sex, Steroid, Antivirals, Fatigue, Malaise,
       Anorexia, Liver_Big, Liver_Firm, Spleen_Palpable,Spiders,
       Ascites , Varices, Bilirubin, Alk_Phosphate, SerumGt,
       Albumin, Protime, Histology]])
    try:
        predTree = hepTree.predict(X_data)
        if predTree == 1:
            print("I'm sorry but you might not make it.Your chances are low")
        else:
            print("Congratulations, you might make it."
                  "Just keep up with your apointments and let the Doctor know if you have anymore symptoms ")
    except:
        print('An error occured')
    next_calc = input("Do you want to perform another Prediction (yes/no)")
    if 'no' in next_calc:
        break


# In[ ]:




