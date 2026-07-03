import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

#Load Titanic Dataset
df=pd.read_csv(r"C:\Users\bawa\Desktop\python.py\titanic.csv")
print(df.head())

#Inspection of dtypes and missingness
print(df.dtypes)
print(df.info())
print(df.describe())
print(df.isnull().sum())

#Survival Rates(Bar Charts)

sns.barplot(x="Sex", y="Survived", data=df)
plt.title("Survival Rate by Sex")
plt.show()

sns.barplot(x="Pclass", y="Survived", data=df)
plt.title("Survival Rate by class")
plt.show()

df["AgeGroup"]= pd.cut(df["Age"], bins=[0,12,18,30,50,80], labels=["Child", "Teen", "Young Adult","Adult","Senior"])
sns.barplot(x="AgeGroup", y="Survived", data=df)
plt.title("Survival Rate by Age Group")
plt.show()

#Boxplot
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age distribution by Survival")
plt.show()

#Violin plot
sns.violinplot(x="Survived", y="Age", data=df)
plt.title("Survival vs Age Distribution")
plt.show()

#ML model
print("=====ML Model=====")
df=df[["Survived","Pclass","Sex","Age","Fare"]]
#Handle missing values
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Fare"]=df["Fare"].fillna(df["Fare"].median())
#Categorical to numaric
df["Sex"]=df["Sex"].map({"male":0, "female":1})
#Target
X=df[["Pclass","Sex","Age","Fare"]]
y=df["Survived"]
#Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
#Model training
model=LogisticRegression(max_iter=200)
model.fit(X_train,y_train)
#Preciction
y_pred=model.predict(X_test)
#Evalution
print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
#Test single prediction
sample=pd.DataFrame({"Pclass":[3],
                     "Sex":[0],
                     "Age":[25],
                     "Fare":[10]})
prediction=model.predict(sample)
print("Survived" if prediction[0] ==1 else "Not Survived")

Insight= f"""==========INSIGHT=========
1. Female passengers had a significantly higher survival rate than male passengers.

2. Passengers traveling in 1st class were more likely to survive than those in 2nd and 3rd class.

3. Younger passengers showed slightly better survival chances compared to older passengers.

4. Fare and passenger class were important factors influencing survival.

5. Missing values in Age and Fare were handled using median imputation.

6. The Sex column was converted into numerical values for machine learning.

7. A Logistic Regression model was trained using Pclass, Sex, Age, and Fare.

8. The model successfully predicted passenger survival outcomes.

9.Model performance was evaluated using Accuracy, Confusion Matrix, and Classification Report.

10.This project demonstrates the complete workflow of Data Cleaning, EDA, Feature Engineering, and Machine Learning Prediction.

Insight saved as insight.txt"""
with open("insight.txt", "w") as file:
    file.write(Insight)