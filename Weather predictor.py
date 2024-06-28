import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# FUNCTIONS

# Check for if the value is an int or a float
def is_int_or_float(val):
  return isinstance(val, (int, float))



# DATA LOADING 

df = pd.read_csv(r"c:\Users\user\Downloads\weatherAUS.csv")
#df = df.sample(n = 20000, random_state=42)


# INITIAL PREPROCESSING

'''INITIAL ANALYSIS: I will be removing all the wind direction features except for WindDir3pm.
WindDir is generally more relevant than windgustdir and later times yield stronger results
than earlier times. Also, since all the features have the same kind of data, it will be very
hard to encode. I will decide whether to drop other fields based on future analysis'''

# Dropping rows with NaN values and duplicate rows
df = df.dropna()
df = df.drop_duplicates()
#print(df)

# Dropping selected features
df = df.drop('WindGustDir', axis=1).drop('WindDir9am', axis=1)
df = df.drop('Location', axis=1).drop('Date', axis=1)

# Label encoding tomorrow rain
nxtdayrain_df = pd.get_dummies(df['RainTomorrow'], drop_first='True')
nxtdayrain_df.columns = ['Tmrw rain']





# DATA VISUALISATION AND ANALYSIS

'''PRE VISUALISATION ANALYSIS: There are several features about certain phenomena at certain times. Based on
common sense, I may only keep those features that are later into the day, as they're generally more
important in predicting the occurence of rain. I may remove Windgust features entirely '''
#print(df.describe())

# Temperature plots
#sns.boxplot(df, x='MinTemp', showmeans=True, showfliers=True)
#plt.show()

#sns.boxplot(df, x='MaxTemp', showmeans=True, showfliers=True)
#plt.show()

'''ANALYSIS: For both temps, their IQR is not very high, but the whiskers are far from the IQR points,
suggesting a high variance. Therefore it might be informative. However, there are extreme outliers for both.
Thus, I might consider scaling or transformation of all numerical data.'''

#sns.barplot(df, x='RainTomorrow', y='MinTemp')
#plt.show()

#sns.barplot(df, x='RainTomorrow', y='MaxTemp')
#plt.show()

'''ANALYSIS: Oddly, higher mintemps result in increased chances of rain, but it's the opposite for maxtemp.
Furthermore, the difference in the two isn't that high, so it probably means temperature alone is insignificant.'''

'''sns.scatterplot(df, x='Sunshine', y='RainTomorrow')
plt.show()

sns.scatterplot(df, x='Evaporation', y='RainTomorrow')
plt.show()'''

'''ANALYSIS: Sunshine was nearly evenly distributed between Yes and No, with the No being very slightly skewed
to the right. So sunshine is probably insignificant on its own. However, evaporation is odd - the Evaporation
points at Yes are of lower values than the ones at No, and those at No are also more spread out and numerous.
This suggests that less evap = more rain, which is counterintuitive, and also suggests that evaporation & sunshine
aren't related. We can test this by removing evaporation and or sunshine.
RESULT: REMOVING EITHER OR BOTH DECREASES ACCURACY.'''



# POST VISUALISATION PROCESSING

# Label encoding

# Loop to label encode each categorical column and drop the original feature later
for col in df.columns:
  if is_int_or_float(df[col].iloc[0]) or col == "RainTomorrow":
    pass
  else:
    encoded_df = pd.get_dummies(df[col], drop_first=True)
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(col, axis=1)


# Loop to transform all numerical values with sqrt to test for improvement
# RESULT:MODEL ACCURACY IMPROVES TO 100%
for col in df.columns:
  if is_int_or_float(df[col].iloc[0]):
    df[col] = np.sqrt(np.abs(df[col]))


df = pd.concat([df, nxtdayrain_df], axis=1).drop('RainTomorrow', axis=1)
df = df.dropna()

# TRAINING
X = df.drop('Tmrw rain', axis=1)
y = df['Tmrw rain']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

'''EVALUATION: Model performs with 100% score post data transformation. Might be due to overfitting.
SIDE Note: Regression Classifier tree gets 100% score without data transformation'''
