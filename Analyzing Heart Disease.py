
# coding: utf-8

# #  Analyze and Predict Heart Disease
#  
# In this session, we will use a Notebook running Python 3.5 with Apache Spark 2.1 for data analysis using Apache SystemML, IBM Cloud Object Storage and pandas DataFrames. We will also use matplotlib and seaborn for visualizations and walk through some examples of Data Analysis, Preparation, Classification, Data Normalization and Correlations. From the Machine Learning perspective, we will go through the ETL where we will train and test or data, run some classifiers to see which one is the best option, feature and model creation using the scikit learn for the Logistic Regression classifier and finaly test our model.
# 
# We will analyze open data from Heart Disease UCI and to extract even more insights, we will explore this dataset, build charts for visualization of specific areas and see how the data science can help predicting heart disease.

# # Install prerequisites
# 
# To start, we will import NumPy, the fundamental library for array computing with Python, pandas for our dataframes, matplotlib and seaborn for visualizations.
# 

# In[1]:


#Import the necessary libraries needed for data exploration and visualization.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Understanding the Dataset
# 
# 
# Data Set Information:
# 
# We will use the open data from Heart Disease UCI Machine Learning Repository. This dataset contains 14 columns and 303 records.
# 
# Attributes Information:
# 
#      age      - age in years
#      sex      - (1 = male; 0 = female)
#      cp       - chest pain type
#      trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#      chol     - serum cholestoral in mg/dl
#      fbs      - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#      restecg  - resting electrocardiographic results
#      thalach  - maximum heart rate achieved
#      exang    - exercise induced angina (1 = yes; 0 = no)
#      oldpeak  - ST depression induced by exercise relative to rest
#      slope    - the slope of the peak exercise ST segment
#      ca       - number of major vessels (0-3) colored by flourosopy
#      thal     - 3 = normal; 6 = fixed defect; 7 = reversable defect
#      target   - have disease or not (1=yes, 0=no) (the predicted attribute)
# 
# 
# About UCI Machine Learning Repository
# 
# The UCI Machine Learning Repository is a collection of databases, domain theories, and data generators that are used by the machine learning community for the empirical analysis of machine learning algorithms.
# 
# Source: https://archive.ics.uci.edu/ml/datasets/Heart+Disease

# # Import the 'heart.csv' dataset into the Notebook
# 
# Before loading the file into your IBM Cloud Object Storage by dragging and dropping the file on the '1001' panel, you need to create a connection in your Notebook to your IBM Cloud Object Storage. To do that, from your project page, click on the 'Add to Project' and then on 'Connection', choose your Cloudant Instance and then click on 'Create'.
# 
# And then we will connect the Notebook to the IBM Cloud Object Storage by Inserting your Credentials and load the "heart.csv" dataset by Inserting pandas DataFrame so we can start with our data analysis.

# In[2]:





# In[3]:


df_data_1.tail()


# # Initial Data Preparation and Exploration
# 

# Checking the shape of the dataset

# In[4]:


df_data_1.shape


# Let's use the Describe function to see the count, mean, std, min, max, 25%, 50%, 75% values

# In[5]:


df_data_1.describe()


# Checking the details of each column

# In[6]:


df_data_1.info()


# Notice that each column name begins with a lower case, let's rename all column names to begin with upper case

# In[7]:


df_data_1=df_data_1.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol','fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang','oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})


# Checking the new column names

# In[8]:


df_data_1.columns


# Now let's look for any null value

# In[9]:


df_data_1.isna().sum()


# Target columns shows if the record have disease or not (1=yes, 0=no) from all dataset:
# 
# Target - have disease or not (1=yes, 0=no) (the predicted attribute)

# In[10]:


df_data_1.Target.value_counts()


# Getting the percentage of the Target column details

# In[11]:


Disease_Count_yes = len(df_data_1[df_data_1.Target == 1])
Disease_Count_no = len(df_data_1[df_data_1.Target == 0])
print("{:.2f}% have heart disease".format((Disease_Count_yes / (len(df_data_1.Target))*100)))
print("{:.2f}% does not have heart disease".format((Disease_Count_no / (len(df_data_1.Target))*100)))


# Now let's plot the details of Target column

# In[12]:


sns.countplot(x="Target", data=df_data_1, palette="Greens_r")
plt.show()


# From the Sex column, we can see the percentage of males and females (1 = male; 0 = female)

# In[13]:


Male_count = len(df_data_1[df_data_1.Sex == 1])
Female_count = len(df_data_1[df_data_1.Sex == 0])
print("{:.2f}% are Male".format((Male_count / (len(df_data_1.Sex))*100)))
print("{:.2f}% are Female".format((Female_count / (len(df_data_1.Sex))*100)))


# And plot the results

# In[14]:


sns.countplot(x='Sex', data=df_data_1, palette="Greens_r")
plt.xlabel("0 = female 1= male")
plt.show()


# Now let's explore the Age column getting the count

# In[15]:


df_data_1.Age.value_counts()[:20]


# And plot the results

# In[16]:


sns.barplot(x=df_data_1.Age.value_counts()[:20].index,y=df_data_1.Age.value_counts()[:20].values)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Age Analysis')
plt.show()


# Analyzing the mean

# In[17]:


df_data_1.groupby('Target').mean()


# Combining the Age with Target columns, we will see the frequency of heart disease based on the age

# In[18]:


pd.crosstab(df_data_1.Age,df_data_1.Target).plot(kind="area",figsize=(20,6))
plt.title('Heart Disease Frequency based on age')
plt.xlabel('Ages')
plt.ylabel('Frequency')
plt.show()


# Also the frequency based on chest pain type

# In[19]:


pd.crosstab(df_data_1.Cp,df_data_1.Target).plot(kind="kde",figsize=(15,6),color=['#11A5AA','#AA1190'])
plt.title('Frequency based on the Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency if disease or Not')
plt.show()


# Male's age average who suffered a stroke

# In[20]:


df_data_1[(df_data_1.Target ==  1) & (df_data_1.Sex == 1)].Age.mean()


# Female's age average who suffered a stroke

# In[21]:


df_data_1[(df_data_1.Target ==  1) & (df_data_1.Sex == 0)].Age.mean()


# Checking the correlation values between them

# In[22]:


df_data_1.corr()


# Correlation Matrix

# In[23]:


fig,ax = plt.subplots(figsize=(20, 8))
sns.heatmap(df_data_1.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="viridis")
plt.show()


# # ETL

# Import all the necessary libraries

# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Creating the array

# In[25]:


y=df_data_1.Target.values
x_data=df_data_1.drop(["Target"],axis=1)


# In[26]:


y


# 
# Data Normalization / feature scaling

# In[27]:


x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# In[28]:


x.head()


# Now we will split or data:
# 
# * 70 % will be train data
# * 30% will be test data

# In[29]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# We will use several classifiers to check which one has the highest accuracy score

# Since we will be using some classifiers, let's store the results in a list for a better view at the end of this session

# In[30]:


scores_accuracy=[]


# ## Decision Tree classifier

# In[31]:


#from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
dt_score = dt.score(x_test,y_test)
scores_accuracy.append(["DT",dt_score])
print("Decision Tree Accuracy: ",dt.score(x_test,y_test))


# In[32]:


#from sklearn.metrics import confusion_matrix
y_predict_dt = dt.predict(x_test)
y_real_dt = y_test
cm_dt = confusion_matrix (y_real_dt,y_predict_dt)
f, ax =plt.subplots(figsize=(3,3))
sns.heatmap(cm_dt,annot=True,linewidths=0.2,linecolor="black",fmt=".0f",cmap="viridis",ax=ax)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()


# ## K-nearest neighbors (KNN) classifier

# In[33]:


#from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
knn_score = knn.score(x_test,y_test)
scores_accuracy.append(["KNN",knn_score])
print(" KNN Accuracy with {} nn: {} ".format(12,knn.score(x_test,y_test)))


# In[34]:


#from sklearn.metrics import confusion_matrix
y_predict_knn = knn.predict(x_test)
y_real_knn = y_test
cm_knn = confusion_matrix (y_real_knn,y_predict_knn)
f, ax =plt.subplots(figsize=(3,3))
sns.heatmap(cm_knn,annot=True,linewidths=0.2,linecolor="black",fmt=".0f",cmap="viridis",ax=ax)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()


# ## Logistic Regression classifier

# In[35]:


#from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(x_train,y_train)
lr_score = lr.score(x_test,y_test)
scores_accuracy.append(["LR",lr_score])
print("Logistic Regression Accuracy: {}".format(lr.score(x_test,y_test)))


# In[36]:


#from sklearn.metrics import confusion_matrix
y_predict = lr.predict(x_test)
y_real = y_test
cm = confusion_matrix (y_real,y_predict)
f, ax =plt.subplots(figsize=(3,3))
sns.heatmap(cm,annot=True,linewidths=0.2,linecolor="black",fmt=".0f",cmap="viridis",ax=ax)
plt.xlabel("Predict")
plt.ylabel("Real")
plt.show()


# Let's compare the score of each algorithm

# In[37]:


scores_accuracy


# From those three classifiers, Logistic Regression has the best accuracy: 0.79120879120879117 and will be used in this session
# 

# In[38]:


from sklearn.model_selection import train_test_split
from pyspark.sql.types import *
x_train, x_test = train_test_split(df_data_1,test_size =0.2,random_state=0)
mySchema = StructType([StructField("Age", IntegerType(), False)                       ,StructField("Sex", IntegerType(), False)                       ,StructField("Cp", IntegerType(), False)                       ,StructField("Trestbps", IntegerType(), False)                       ,StructField("Chol", IntegerType(), False)                       ,StructField("Fbs", IntegerType(), False)                       ,StructField("Restecg", IntegerType(), False)                       ,StructField("Thalach", IntegerType(), False)                       ,StructField("Exang", IntegerType(), False)                       ,StructField("Oldpeak", FloatType(), False)                       ,StructField("Slope", IntegerType(), False)                       ,StructField("Ca", IntegerType(), False)                       ,StructField("Thal", IntegerType(), False)                       ,StructField("Target", IntegerType(), False)])
df_data_1_train = spark.createDataFrame(x_train, schema=mySchema)
df_data_1_test = spark.createDataFrame(x_test, schema=mySchema)


# ## Feature Creation

# In[39]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

indexer = StringIndexer(inputCol="Target", outputCol="label")
vectorAssembler = VectorAssembler(inputCols=["Age","Sex","Cp","Trestbps","Chol","Fbs","Restecg","Thalach","Exang","Oldpeak","Slope","Ca","Thal"],outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)


# ## Model Creation - Final Model

# In[40]:


from pyspark.ml.classification import LogisticRegression

log_reg = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)


# In[41]:


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=(indexer, vectorAssembler, normalizer, log_reg))


# In[42]:


model=pipeline.fit(df_data_1_train)


# In[43]:


prediction = model.transform(df_data_1_train)


# ## Evaluation

# In[44]:


from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluation = MulticlassClassificationEvaluator().setMetricName('accuracy').setLabelCol('label').setPredictionCol('prediction')


# In[45]:


evaluation.evaluate(prediction)


# ## Test

# In[46]:


model=pipeline.fit(df_data_1_test)


# In[47]:


prediction = model.transform(df_data_1_test)


# In[48]:


evaluation.evaluate(prediction)


# # Conclusion
# 
# After running various classifiers, the Logistic Regression model had the best performance. But if you want to build a larger project, you could also run other classifiers such as Random Forest, Gradient Boosting, KNN, SVM and Naive Bayes to go deeper on your analysis.
# 
# 
# The main indicators of heart disease stem from the healthiness of your heart (seems quite obvious). Although other health indicators like resting bps and blood sugar are good for overall health. Heart disease stems from having an unhealthy heart
