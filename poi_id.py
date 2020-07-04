#!/usr/bin/env python
# coding: utf-8

# # Enron Case-Person of Interest

# In[1]:


#!/usr/bin/python
## Author is Rajdeep
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
##  tester.py is modified becuase of renaming and deprecation of cross_validation sub-module to model_selection
from tester import dump_classifier_and_data
##  tester.py is modified becuase of renaming and deprecation of cross_validation sub-module to model_selection


# In[2]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list=['poi','salary']


# In[3]:


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[4]:


data_dict


# In[5]:


### Data exploration
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict


# In[6]:


#Number of suspects
name_data_point=data_dict.keys()


# In[7]:


print 'Total number of datapoints are {}'.format(len(name_data_point))


# In[8]:


#Number of features
total_features=len (data_dict[name_data_point[0]])
print 'Total number of features are {}'.format(total_features)


# In[9]:


features_temp=data_dict[name_data_point[0]].keys()


# In[10]:


#For convinience of data cleaning (Removing NaN) using pandas (Ref: Data visulization)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[11]:


unwanted_features=["poi","email_address"]

features_temp=[ele for ele in features_temp if ele not in unwanted_features]
# to make the first element in the list "poi"
features_temp=["poi"]+features_temp
feature_data= featureFormat(data_dict,features_temp,remove_NaN=False)


# In[12]:


temp_df=pd.DataFrame(data=feature_data,columns=features_temp, index=name_data_point)
print "With NaN"
print temp_df.info()


# In[13]:


poi=temp_df['poi']==1
temp_df[poi].count()


# In[14]:


temp_df=temp_df.fillna(0)
print "Removing NaN and replacing with 0"
print temp_df.info()


# ## Outlier detection and data cleaning 

# In[15]:


# Determining the outliers for financial datas
financial_features=["salary", "total_payments","long_term_incentive","exercised_stock_options","bonus","restricted_stock", "total_stock_value", "expenses",'deferred_income',"deferral_payments"]
temp_df[financial_features].describe()


# In[16]:


temp_df[temp_df["restricted_stock"]<0]


# In[17]:


temp_df.info()


# In[18]:


temp_df[temp_df["salary"]==temp_df["salary"].max()]
# TOTAL seems odd, and it was pointed out in the session that this could be a human error hence this a outlier too.


# In[19]:


temp_df=temp_df.drop(index="TOTAL",axis=0)


# In[20]:


temp_df[financial_features].loc[(temp_df[financial_features]==0).all(axis=1)].index
# found this one outliers with all finantial features as zero, hence it can be dropped


# In[21]:


# Dropping Lockhart from data set
temp_df=temp_df.drop(temp_df[financial_features].loc[(temp_df[financial_features]==0).all(axis=1)].index,axis=0)


# In[22]:


temp_df.info()


# In[23]:


temp_df.loc['THE TRAVEL AGENCY IN THE PARK'].index


# In[24]:


temp_df=temp_df.drop(['THE TRAVEL AGENCY IN THE PARK'],axis=0)
temp_df.info()


# In[25]:


#converting back to dict

data_dict= temp_df.to_dict('index')
features_list=list(temp_df.columns.values)
features_list


# In[26]:


# Reference module from feature selection: Used for creating new features "fraction_to_poi" and "fraction_from_poi"
def computeFraction( poi_messages, all_messages ):

    if poi_messages!=0 and all_messages!=0:
        fraction = float(poi_messages)/float(all_messages)
        return fraction
    else:
        fraction = 0.0
    



        return fraction


# In[27]:


# Used for creating new features "total_profit"
def total_profit(bonus, salary,long_term_incentive):
    profit=float(bonus)+float(salary)+float(long_term_incentive)
    
    return profit
    


# In[28]:


submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    
    data_point["fraction_from_poi"] = fraction_from_poi
    
    bonus=data_point["bonus"]
    salary=data_point["salary"]
    long_term_incentive=data_point["long_term_incentive"]


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    profit=total_profit(bonus, salary,long_term_incentive)
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi,
                      "total_profit":total_profit}
    data_point["fraction_to_poi"] = fraction_to_poi
    data_dict[name]["fraction_to_poi"]=fraction_to_poi
    data_dict[name]["fraction_from_poi"]= fraction_from_poi
    data_dict[name]["total_profit"]=profit


# In[29]:


#added two additional features in data_dict
total_features=len (data_dict[name_data_point[0]])
print 'After cleaning Total number of features are {}'.format(total_features)


# In[30]:


print 'After cleaning number of data points {}'.format(len(data_dict))


# In[31]:


dict_cols=list(data_dict[name_data_point[0]].keys())


# In[32]:


my_dataset = data_dict
for names in dict_cols:
    if names not in features_list:
        print names
        features_list.append(names)

features_list


# ## SVC Classifier with PCA and MinMaX Scalar

# In[178]:


#With all features
from time import time
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[179]:


data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test =train_test_split(features, labels, test_size=0.3, random_state=42)


# In[180]:


from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.model_selection import StratifiedShuffleSplit
clf_svc=SVC()
pca = RandomizedPCA(n_components=11, whiten=True).fit(features_train)
ss = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state = 42)
param_grid_svc={'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}
grid=GridSearchCV(clf_svc,param_grid_svc,cv=ss, scoring = 'f1')


# In[181]:


t0=time()
grid.fit(features_train,labels_train)
print "Training time for regular SVC with default parameters is %0.3fs" % (time() - t0)


# In[182]:


grid.best_estimator_


# In[183]:


y_pred=grid.predict(features_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print "CLASSIFICATION REPORT of SVC "
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX of SVC "
print confusion_matrix(labels_test, y_pred, labels=[0,1])
print "--------The f1-score of SVC classifer------: "
print f1_score(labels_test,y_pred)


# In[187]:


#Regular SVC without default parameters

scaler = MinMaxScaler().fit(features_train)

#Minmax transform
X_train_scaler =  scaler.transform(features_train)
X_test_scaler =scaler.transform(features_test)
### PCA transform
X_train_pca = pca.transform(features_train)
X_test_pca = pca.transform(features_test)
## PCA and scaler
X_train_pca_scaler=pca.transform(X_train_scaler)
X_test_pca_scaler=pca.transform(X_test_scaler)


# In[188]:


t0=time()
grid.fit(X_train_pca,labels_train)
print "Training time for regular SVC with PCA is %0.3fs" % (time() - t0)


# In[190]:


grid.best_estimator_


# In[126]:


y_pred=grid.predict(X_test_pca)
print "CLASSIFICATION REPORT of SVC with PCA "
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX of SVC with PCA"
print confusion_matrix(labels_test, y_pred, labels=[0,1])
print "--------The f1-score of SVC classifer with PCA------: "
print f1_score(labels_test,y_pred)


# In[127]:


t0=time()
grid.fit(X_train_scaler,labels_train)
print "Training time for regular SVC with scaler is %0.3fs" % (time() - t0)


# In[128]:


y_pred=grid.predict(X_test_scaler)
print "CLASSIFICATION REPORT of SVC with scaler  "
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX of SVC with with scaler"
print confusion_matrix(labels_test, y_pred, labels=[0,1])
print "--------The f1-score of SVC with scaler------: "
print f1_score(labels_test,y_pred)


# ## Naive Bayes Gaussian

# In[129]:


from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
clf_gnb=GaussianNB()
#Using SKbest module to choose best number of features for classification
skb=SelectKBest()


# In[130]:


from sklearn.pipeline import Pipeline
pipe_2=Pipeline([('SKB',skb),('GNB', clf_gnb)])

pipe_2.get_params().keys()


# In[131]:


param_grid={"SKB__k":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}
clf_grid_gnb=GridSearchCV(pipe_2,param_grid,cv=ss, scoring = 'f1')


# In[132]:


t0=time()
clf_grid_gnb.fit(features,labels)
print "training done for GNB in %0.3fs" % (time() - t0)


# In[133]:


clf_grid_gnb.best_estimator_


# In[134]:


clf=clf_grid_gnb.best_estimator_
clf.fit(features_train,labels_train)
y_pred=clf.predict(features_test)


# In[135]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print "CLASSIFICATION REPORT for Gaussian"
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX for Gaussian"
print(pd.DataFrame(confusion_matrix(labels_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))


# In[136]:


print "---------Accuracy of GNB------------"
print accuracy_score(labels_test, y_pred)
print "\n"
print "-------------Precision of GNB ----------- "
print precision_score(labels_test,y_pred)
print "\n"
print "----------Recall of  GNB -----------    "
print recall_score(labels_test,y_pred)
print "\n"
print "--------The f1-score of GNB------: "
print f1_score(labels_test,y_pred)


# ## Adaboost

# In[156]:


from sklearn.ensemble import AdaBoostClassifier
clf_ada = AdaBoostClassifier(random_state=0)
pipe_ada=Pipeline([('SKB',skb),('ADA', clf_ada)])
pipe_ada.get_params().keys()


# In[157]:


param_grid={"SKB__k":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'ADA__algorithm':['SAMME','SAMME.R'] }
clf_grid_ada=GridSearchCV(pipe_ada,param_grid,cv=ss, scoring = 'f1')

clf_grid_ada.fit(features,labels)


# In[159]:


clf_ada=clf_grid_ada.best_estimator_

clf_ada.fit(features_train,labels_train)
y_pred=clf_ada.predict(features_test)


# In[160]:


print "CLASSIFICATION REPORT for Trees"
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX for Trees"
print(pd.DataFrame(confusion_matrix(labels_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))


# In[161]:


print "---------Accuracy of ADABOOST------------"
print accuracy_score(labels_test, y_pred)
print "\n"
print "-------------Precision of ADABOOST ----------- "
print precision_score(labels_test,y_pred)
print "\n"
print "----------Recall of  ADABOOST -----------    "
print recall_score(labels_test,y_pred)
print "\n"
print "--------The f1-score of ADABOOST------: "
print f1_score(labels_test,y_pred)


# ## Decision Trees

# In[191]:


from sklearn.tree import DecisionTreeClassifier
clf_tree=DecisionTreeClassifier(random_state=0)
t0=time()
clf_tree.fit(features_train,labels_train)
print "DT training done for regular tree in %0.3fs" % (time() - t0)


# In[192]:


y_pred=clf_tree.predict(features_test)


# In[193]:


print "CLASSIFICATION REPORT"
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX"
print(pd.DataFrame(confusion_matrix(labels_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))


# In[194]:


print "---------Accuracy of Decision Tree Default------------"
print accuracy_score(labels_test, y_pred)
print "\n"
print "-------------Precision of Decision Tree Default ----------- "
print precision_score(labels_test,y_pred)
print "\n"
print "----------Recall of  Decision Tree Default -----------    "
print recall_score(labels_test,y_pred)
print "\n"
print "--------The f1-score of Decision Tree Default------: "
print f1_score(labels_test,y_pred)


# In[165]:


pipe_tree=Pipeline([('SKB',skb),('tree', clf_tree)])
pipe_tree.get_params().keys()


# In[166]:


param_grid={"SKB__k":[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'tree__criterion':['entropy'],'tree__min_samples_split':[2,5,7,10,12,15],'tree__max_depth':[1,2,3,4,5]}
clf_grid_tree=GridSearchCV(pipe_tree,param_grid,cv=ss, scoring = 'f1')


# In[167]:


t0=time()
clf_grid_tree.fit(features,labels)
print "training done for Tree in pipeline in %0.3fs" % (time() - t0)


# In[168]:


clf_grid_tree.best_estimator_


# In[173]:


clf_tree=clf_grid_tree.best_estimator_
clf_tree.fit(features_train,labels_train)
y_pred=clf_tree.predict(features_test)


# In[174]:


print "CLASSIFICATION REPORT for Trees"
print classification_report(labels_test, y_pred)
print "CONFUSION MATRIX for Trees"
print(pd.DataFrame(confusion_matrix(labels_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))


# In[175]:


print "---------Accuracy of Decision Tree------------"
print accuracy_score(labels_test, y_pred)
print "\n"
print "-------------Precision of Decision Tree ----------- "
print precision_score(labels_test,y_pred)
print "\n"
print "----------Recall of  Decision Tree -----------    "
print recall_score(labels_test,y_pred)
print "\n"
print "--------The f1-score of Decision Tree------: "
print f1_score(labels_test,y_pred)


# In[177]:


dump_classifier_and_data(clf_tree, my_dataset, features_list)


# In[ ]:




