#!/usr/bin/env python
# coding: utf-8

# # Previous settings
# ### This section contains 2 cells. The packages used later are imported. The routes to read the dataset, import models and export predicted results are defined.

# In[17]:


from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib
import pandas as pd
import os 

# Suppress the warning caused by sklearn
import warnings
warnings.filterwarnings('ignore')


# In[18]:


# Setting the route to read the data set
PROJECT_ROOT_DIR = "."
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, "datasets")
os.makedirs(DATA_PATH, exist_ok=True)

# Setting the route to save the result of prediction
RESULT_PATH = os.path.join(PROJECT_ROOT_DIR, "results")
os.makedirs(RESULT_PATH, exist_ok=True)

# Setting the route to load the model
MODEL_PATH = os.path.join(PROJECT_ROOT_DIR, "models")
os.makedirs(MODEL_PATH, exist_ok=True)


# # Data reading and preprocessing
# ### This section contains 2 cells. The dataset ‘dataset2.xlsx’, consisting of MORB glasses with complete major and trace elements but without determined water contents, is imported from the secondary directory ‘datasets’. The needed element compositions are extracted to ‘X_all’.

# In[19]:


# Read the excel of the data
data = os.path.join(DATA_PATH,"dataset2.xlsx")
df1 = pd.read_excel(data)
df1


# In[20]:


X_all = df1
X_all


# # Water predicting
# ### This section contains 4 cells. The pre-established model ‘Established_RFR_model.pkl’ is imported from the secondary directory ‘models’. The water contents of samples are predicted by the model and inserted into the dataset with a column name of ‘H2O_P’. The predicted results are exported to file ‘result.xlsx’ in the secondary directory ‘results’. 
# ### It is noteworthy that if users have trained a new model in the first file and want to predict samples by the new model, a transfer or copy of the new model from the secondary directory ‘models’ in the first file to the secondary directory ‘models’ in the second file is recommended, and the newly-trained model names in the code and in the secondary directory should be consistent to make sure the code can import the model successfully.

# In[21]:


# load the trained model "Established_RFR_model.pkl"
#model = os.path.join(MODEL_PATH,"Established_RFR_model.pkl")
#获取上一级路径
#dirs = os.path.abspath(os.path.dirname(os.getcwd())+"/model_training_code/models/")
#model_path = os.path.join(dirs, "rf.pkl")
model_path ="C:/Users/Leo/Desktop/MORBglass_prediction_2022_JingjunZhou-main/model_training_code/models"
model_rf = joblib.load(model_path)


# In[22]:


result_rf = model_rf.predict(X_all)
np.array(result_rf,dtype=float)
result_rf


# In[23]:


# Insert the predicted H2O contents into the input dataset
df1.insert(df1.shape[1], 'H2O_P', result_rf)
df1


# In[24]:


# Output results
result = os.path.join(RESULT_PATH,"result.xlsx")
df1.to_excel(result,"result.xlsx")

