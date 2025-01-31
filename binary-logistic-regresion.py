import numpy as np

import pandas as pd

from  sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score , confusion_matrix

from sklearn.metrics import roc_curve

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV



data = pd.read_csv("train.csv")

data.drop(['Name' , 'Ticket' , 'Cabin' , 'PassengerId','Fare'] , axis=1 ,inplace=True)

data['Age'] = data['Age'].fillna(data['Age'].median())

data['Sex'] = data['Sex'].map({'male':0 , 'female': 1})

#data['Embarked'] = data['Embarked'].map({'S':0 , 'C':1 , 'Q':2})

data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data = pd.get_dummies(data , columns=['Embarked'] , drop_first=True)

data[['Embarked_Q' , 'Embarked_S']] = data[['Embarked_Q' , 'Embarked_S']].astype(int)

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S', 'Survived']

columns2 =['Age', 'SibSp', 'Parch']



data[['Age', 'SibSp', 'Parch']] =np.log1p(data[['Age', 'SibSp', 'Parch']])  


figure_data = data.melt(value_vars=columns , var_name='features' , value_name='value')

plt.figure(figsize=(12, 6))

sns.boxplot(x = 'features' , y = 'value' , data=figure_data)

plt.title("To see the outlier")

fig, axes = plt.subplots(2, 4, figsize=(15, 8))  

axes = axes.flatten()  

for i, col in enumerate(columns):
    
    sns.boxplot(y=data[col], ax=axes[i])

    axes[i].set_title(col)

plt.tight_layout()



x = data[['Pclass' , 'Sex' , 'Age' , 'SibSp' , 'Parch' ,'Embarked_Q' , 'Embarked_S']]

y = data['Survived']

x_train , x_test , y_train , y_test = train_test_split(

    x , y , test_size=0.2 , random_state=42

)


param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

model = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000), param_grid, cv=5)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


print(classification_report(y_test , y_pred))

conf_matr = confusion_matrix(y_test , y_pred)


plt.figure(figsize=(6,4))

sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.title('Confusion Matrix')

plt.figure(figsize=(6,4))

fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test)[:,1])

plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]):.2f}")

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()