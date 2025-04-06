# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:59:17 2025

@author: richi
"""
# importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# loading the data
data = pd.read_csv(r"C:\Users\richi\Downloads\Social_Network_Ads.csv")
#copying the data
df = data.copy()
# descriptive satistics
df.shape
df.columns
df.info()
df.isnull().sum()
df.describe()

# plotting the data distribution
sns.kdeplot(df['Age'])
plt.title('dist_of_age')
plt.show()

sns.kdeplot(df['EstimatedSalary'])
plt.title('dist_of_EstimatedSalary')
plt.show()

#encoding the categorical variables as per the data
def cat_encoded_var(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df

df = cat_encoded_var(df)
       
#dividing the data btw x and y 
X = df.iloc[:,1:4]
y = df.iloc[:,-1]        
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

#scalling the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)       

#logistic regression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
model_LR = LogisticRegression()
model_LR.fit(X_train, y_train)
y_pred_lr = model_LR.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_lr))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

model_LR_CV = LogisticRegressionCV(cv=100 ,max_iter=2000, random_state=42)
model_LR_CV.fit(X_train, y_train)
y_pred_lr_cv = model_LR_CV.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_lr_cv))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_lr_cv))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr_cv))

# decesion tree 
from sklearn.tree import DecisionTreeClassifier
model_DT = DecisionTreeClassifier()
model_DT.fit(X_train,y_train)
y_pred_DT = model_DT.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_DT))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_DT))
print("\nClassification Report:\n", classification_report(y_test, y_pred_DT))

# randomforest classifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
model_RFC = RandomForestClassifier()
model_RFC.fit(X_train,y_train)
y_pred_RFC = model_RFC.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_RFC))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_RFC))
print("\nClassification Report:\n", classification_report(y_test, y_pred_RFC))

# adaboost classifier
model_ABC = AdaBoostClassifier(n_estimators=200,learning_rate=0.001)
model_ABC.fit(X_train,y_train)
y_pred_ABC = model_ABC.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_ABC))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_ABC))
print("\nClassification Report:\n", classification_report(y_test, y_pred_ABC))

# gradient boosting classifier
model_GBC = GradientBoostingClassifier()
model_GBC.fit(X_train,y_train)
y_pred_GBC = model_GBC.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_GBC))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_GBC))
print("\nClassification Report:\n", classification_report(y_test, y_pred_GBC))

# XGBoost classifier
from xgboost import XGBClassifier
model_XGBC = XGBClassifier()
model_XGBC.fit(X_train,y_train)
y_pred_XGBC = model_GBC.predict(X_test)
print("Accuracy:\n", accuracy_score(y_test, y_pred_XGBC))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_XGBC))
print("\nClassification Report:\n", classification_report(y_test, y_pred_XGBC))

# ROC 
from sklearn.metrics import roc_curve, auc
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Adaboost":AdaBoostClassifier(n_estimators=200,learning_rate=0.001),
    "Decesion Tree ":DecisionTreeClassifier()
}

plt.figure(figsize=(10, 7))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Plot random line
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guess')

# Plot details
plt.title('ROC Curves of Multiple Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()















