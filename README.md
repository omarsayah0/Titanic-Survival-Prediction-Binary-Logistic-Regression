# Titanic

## About
This project predicts passenger survival on the Titanic using **Logistic Regression**.  
It demonstrates a complete end-to-end machine learning workflow including **data preprocessing**, **feature engineering**, **hyperparameter tuning** with `GridSearchCV`, and performance evaluation using **Confusion Matrix** and **ROC Curve**.

---

## Files
- `binary-logistic-regresion.py` → Main Python script implementing the Logistic Regression model.  
- `train.csv` → Dataset containing passenger information such as age, gender, class, and survival status.

---

## Steps Included

### 1️⃣ Data Preprocessing
- Loaded the Titanic dataset using `pandas`.  
- Dropped irrelevant columns: `Name`, `Ticket`, `Cabin`, `PassengerId`, and `Fare`.  
- Filled missing `Age` values with the **median**.  
- Encoded `Sex` as numeric (`male = 0`, `female = 1`).  
- Handled missing `Embarked` values using the **mode**.  
- Applied **One-Hot Encoding** to the `Embarked` column (`Embarked_Q`, `Embarked_S`).  
- Applied **log transformation** to reduce skewness for:  
  `Age`, `SibSp`, and `Parch`.

### 2️⃣ Feature Engineering
- Selected features:  
  `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked_Q', 'Embarked_S']`
- Target variable:  
  `['Survived']`

### 3️⃣ Model Training
Used **Logistic Regression** as the classifier.  
Performed **Grid Search Cross Validation** to find the best parameters.

`GridSearchCV` parameter grid:
   ```python 
   param_grid = {
     'C': [0.001, 0.01, 0.1, 1, 10, 100],
     'penalty': ['l1', 'l2']
   }
  ```

### 4️⃣ Evaluation

- Evaluated model performance using:

  - classification_report

  - confusion_matrix

  - ROC AUC Score

- Plotted:

  - Confusion Matrix Heatmap

  - ROC Curve

### 5️⃣ Libraries Used

#### - numpy : Used for mathematical transformations and numerical operations.
###### Example:
```python 
data[['Age', 'SibSp', 'Parch']] = np.log1p(data[['Age', 'SibSp', 'Parch']])
```

### - pandas : Used to load and preprocess the dataset (drop columns, handle missing values, encode categories).
###### Example:
```python 
data = pd.read_csv("train.csv")
data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Fare'], axis=1, inplace=True)
```

### - scikit-learn : Used for model training, hyperparameter tuning, data splitting, and evaluation metrics.
###### Example:
```python 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, classification_report

# Train-test split:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model training with GridSearchCV:
model = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000), param_grid, cv=5)
model.fit(x_train, y_train)
```

### - seaborn : Used to create visualizations for data distribution and model evaluation.
###### Example:
```python 
sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
```

### - matplotlib : Used to plot graphs like boxplots and ROC curves.
###### Example:
```python 
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, model.predict_proba(x_test)[:,1]):.2f}")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
```

## How to Run

1. Install Dependencies :
   ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn

2. Run :
   ```bash
     python binary-logistic-regresion.py

3. Example Output:

<img width="1862" height="1033" alt="image" src="https://github.com/user-attachments/assets/b72cce8a-298d-458e-912f-cea956611fd5" />
<img width="1859" height="987" alt="image" src="https://github.com/user-attachments/assets/025247d6-d1a9-4dc1-ba38-279ab7b36c6d" />
<img width="1118" height="880" alt="image" src="https://github.com/user-attachments/assets/969ec320-d87d-44cf-8b39-10924e58f310" />
<img width="1202" height="880" alt="image" src="https://github.com/user-attachments/assets/1aa9d844-52d8-4dff-8897-f0d3993ca545" />


## Author

Omar Alethamat

LinkedIn : https://www.linkedin.com/in/omar-alethamat-8a4757314/

## License

This project is licensed under the MIT License — feel free to use, modify, and share with attribution.
