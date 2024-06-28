import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
#from sklearn import svm, preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


# DATA LOADING AND CLEANING
og_df = pd.read_csv(r"C:\Users\user\Downloads\Linreg Salary_data.csv")

# Replacing "X Degree with X and fixing misspelling to avoid redundancy"
og_df['Education Level'].replace("Bachelor's Degree", "Bachelor's", inplace=True) 
og_df['Education Level'].replace("Master's Degree", "Master's", inplace=True)
og_df['Education Level'].replace('phD', 'PhD', inplace=True)

# Removing all entries of gender being Other (too small sample size)
og_df = og_df[og_df['Gender'] != "Other"]

df = og_df[['Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']]
df.columns = ['Gender', 'Education', 'Job','YOE', 'Salary']


# Handling of NaN or 0 values
df.dropna(inplace=True)

# Handling of duplicate rows
df.drop_duplicates(inplace=True)

# Removing rows where salary is too low
df = df[df['Salary'] >= 10000]


# VISUALISATION PRE TRAINING

# Visualising gender against salary
#sns.lmplot(df, x='YOE', y='Salary', hue='Gender', ci=None)
#plt.show

# Visualising education level against salary
#sns.lmplot(df, x='YOE', y='Salary', hue='Education', ci=None)
#plt.show

# Plotting Years of experience against Salary
#sns.lmplot(df, x='YOE', y='Salary', ci=None)
#plt.show()

# Bar plot of gender against salary (shows median for both columns)
'''sns.barplot(df, x='Gender', y='Salary', hue='Gender', ci=None)
plt.show()

# Bar plot of education level against salary 
sns.barplot(df, x='Education', y='Salary', hue='Education', ci=None)
plt.show()

# Bar plot to show the counts of each gender
sns.countplot(df, x='Gender', hue='Gender')
plt.show()

# Bar plot to show the counts of each education level
sns.countplot(df, x='Education', hue='Education')
plt.show()'''




# Encoding dummy variables and modifying df
gender_df = pd.get_dummies(df['Gender'], drop_first=True)
edu_df = pd.get_dummies(df['Education'], drop_first=True)
job_df = pd.get_dummies(df['Job'], drop_first=True)
df = pd.concat([gender_df, df], axis=1)
df = pd.concat([edu_df, df], axis=1)
df.drop('Education', axis=1, inplace=True)
df.drop('Gender', axis=1, inplace=True)
df.drop('Job', axis=1, inplace=True)



'''ANALYSIS
Job title might have a large bearing, but since 
there are too many titles to reasonably analyse. So we can remove it.

Education level has a somewhat slight, but noticeable effect on salary.
Typically, High School < Bachelor's < Masters < PhD. However, there is very less PhDs compared
to Bachelors and Masters, so it might not be a completely accurate assessment.

Gender has a bearing as well, but it might not be completely accurate compared to 
Education. Males earn more than females based on the regression line, but that might
be because the outliers in salary were all male (near 250k).

Age and YOE have strong correlation, and both have a generally positive relatonship
with salary. YOE has slightly higher correlation with salary, so we'll remove age.

Thus, with training, we will train using YOE, gender and experience.

I tried data scaling, but since it yielded no improvements while making my code
clunkier, I removed it.'''


# TRAINING

# Initialising X and y values and splitting them
X = df.drop('Salary',axis=1)
y = df['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


# Initialising regression instance
model = RandomForestRegressor(n_estimators=300, max_depth=20)
model.fit(X_train, y_train)

# R2 and MSE values
print("The regression score is", 100 * model.score(X_test, y_test),"%")
y_pred = model.predict(X_test)
print("The MAE is ", mean_absolute_error(y_test, y_pred))
print("The MSE is ", mean_squared_error(y_test, y_pred))
print("The R2 value is",model.score(X_test, y_test))


'''Note: The graphs this time will appear all at once, for some reason'''

 
