import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold

# LOADING DATA
df = pd.read_csv(r"c:\Users\user\Downloads\archive (6)\Social_Network_Ads.csv")
df.columns = ['Age', 'Salary', 'Purchased']
# INITIAL PREPROCESSING

# Removal of null and duplicate observations
df = df.dropna()
df = df.drop_duplicates()

# print(df)
# print(df.describe())

# VISUALISATION

# Plotting age against purchase status
sns.scatterplot(df, x='Age', y='Purchased')
#plt.show()

sns.scatterplot(df, x='Salary', y='Purchased')
#plt.show()

sns.lmplot(df, x='Age', y='Salary')
#plt.show()

'''ANALYSIS: Based on scatterplots, higher salary typically tends to the product being purchased.
The salary range most dense in the No part is 10k to 90k. This suggests that past that, people are
more likely to buy the product. Interestingly enough, the salary range between 20k to 50k is also
somewhat dense in the Yes part. It is probably a special subset of the group in that salary range.
For age, those who are older are more likely to buy the product, but there's a huge overlap. It's more
like past a certain age, some people will buy the product.
There is no correlation between salary and age, meaning that both must be considered when predicting
purchase status. Therefore we will remove nothing.'''


# TRAINING

X = df.drop('Purchased', axis=1)
y = df['Purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
model = RandomForestClassifier(n_estimators=100, max_depth=7, max_features=9, random_state=0, oob_score=True)
model.fit(X_train, y_train)


# EVALUATION 

y_pred = model.predict(X_test)
matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy_score = (matrix[0, 0] + matrix[1, 1])/(matrix[0, 1] + matrix[1, 0] + matrix[0, 0] + matrix[1, 1])
oob = model.oob_score_
print("The accuracy score is", 100* accuracy_score, "%")
print("The OOB score is", oob)
kfold = StratifiedKFold(n_splits=10)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
score = scores.mean()
print("The average accuracy score is", 100 * score, "%")