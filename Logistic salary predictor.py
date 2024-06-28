import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report


# DATA LOADING
df = pd.read_csv(r"C:\Users\user\Downloads\Salary_data.csv")
df.columns = ['YOE', 'Salary']

'''Since there's no empty or duplicate rows, cleaning doesn't need to be done'''

# PREPROCESSING

# Categorising salaries into high salary and low salary
df['High salary'] = df['Salary'] >= 100000
df['Low salary'] = df['Salary'] < 100000


# DATA VISUALISATION

# Visualising years of experience against salary (linear regression graph)
sns.lmplot(df, x='YOE', y='Salary', ci=None)
plt.show()

# Visualising the number of high and low salary employees via bar plot
sns.countplot(df, x='High salary')
plt.show()
sns.countplot(df, x='Low salary')
plt.show()

# Visualising the years of experience against salary status (high or low) via scatterplot
plt.scatter(x=df['YOE'], y=df['High salary'])
plt.show()


# TRAINING

# Splitting the data
X = np.array(df['YOE']).reshape(-1, 1)
y = df['High salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Initialising logistic regression instance
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluating model
y_pred = model.predict(X_test)
print("The model score is", 100 * model.score(X_test, y_test), "%")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# POST TRAINING VISUALISATION
# Plotting years of experience against high salary status as a logistic regression graph
plt.plot(X, y)
plt.xlabel("Years of Experience")
plt.ylabel("Probability of High Salary")
plt.show()