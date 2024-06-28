import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV


# DATA LOADING

df = pd.read_csv(r"RELIANCE.csv")

# INITIAL DATA PREPROCESSING
df = df.dropna().drop_duplicates()
#print(df.describe())

'''ANALYSIS: Prices all have very similar stddev, and their max and min values aren't that
far from the IQR. Leading me to believe that there are not very strong outliers, so 
transformation of price probably isn't needed. We can verify this with plotting boxplots
of the prices. Volume has stronger outliers probably, but verification is also required.'''


'''Since there's no categorical features, there's no need to do label encoding.
'''

# VISUALISATION

# Plotting date against closing price
#sns.lineplot(df, x= 'Date', y='Close')
#plt.show()

'''Based on lineplot analysis, date isn't correlated to the stock price alone (obviously).
There are periods where the closing price peaks, and there are other periods in between where
there's a slump. It's cyclical. Other factors might affect the closing price.'''

# Plotting boxplots for prices
#sns.boxplot(df, x='Close', showmeans=True, showfliers=True)
#plt.show() # slightly longer right whisker, extremely weak skew if any

#sns.boxplot(df, x='Open', showmeans=True, showfliers=True)
plt.show() # equal length whiskers

#sns.boxplot(df, x='High', showmeans=True, showfliers=True)
plt.show() # longer right whisker, might show slight skew on right side

#sns.boxplot(df, x='Low', showmeans=True, showfliers=True)
plt.show() # somewhat longer whisker on right, same as high

'''BOXPLOT ANALYSIS: Overall all the box plots are similar, with a similar IQR, and weak or no outliers.
Skew might be present in high or low, but needs to be investigated further, and probably not 
strong enoug to warrant normalization.'''

# Plotting histograms for low and high prices to observe skew if any
#sns.histplot(df, x='High')
plt.show()
#sns.histplot(df, x='Low')
plt.show()

'''HISTOGRAM ANALYSIS: Contrary to boxplot, histplots show a slight left side skew for both high
and low prices. However, it isn't very much, with low being more skewed. Transformation might
be needed for that.'''

# Plotting graphs to observe correlations between variables.

#sns.lmplot(df, x='Volume', y='Close')
sns.boxplot(df, x='Volume')
#plt.show()

'''I plotted all graphs while changing the lmplot line to save space for code.
Volume appears to have several outliers, as well as a heavy skew to the left, so 
transformation might be necessary. '''

# POST VISUALISATION PREPROCESSING

# Loop for log transformation (We will test with and without)
'''for feature in df.columns:
    if feature != "Date":
        df[feature] = np.log(df[feature])'''





#sns.lmplot(df, x = 'Volume', y='Close')
#plt.show()

# TRAINING

# Splitting data

X = df.drop('Date', axis=1).drop('Close', axis=1)
y = df['Close']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# Training random forest regressor
rfr = RandomForestRegressor(n_estimators=99, max_depth=13, max_features=31, random_state=0, oob_score=True)
rfr.fit(X_train, y_train)


# Evaluating random forest regressor'''
#print("The regression score of RFR is ", 100 * rfr.score(x_test, y_test), "%")
y_pred = rfr.predict(x_test)
print("The OOB score of RFR is", rfr.oob_score_)
print("The MSE of RFR is", mean_squared_error(y_test, y_pred))
print("The MAE of RFR is", mean_absolute_error(y_test, y_pred))
print("The RMSE of RFR is", np.sqrt(mean_squared_error(y_test, y_pred)))


kfold = KFold(n_splits=10, shuffle=False)
kfoldscore_rfr = cross_val_score(rfr, X, y, cv=kfold, scoring='r2').mean()
print("The KFold score of RFR is", 100 * kfoldscore_rfr, "%")


'''OBSERVATION: With no transformation, regression score is higher, but OOB score is worse.
MSE is also higher without transformation, but that's irrelevant as the transformation 
decreases the scale by a lot.
CONCLUSION: We will move forward with no transformation, for SVR As well.'''


# Training Decision tree
tree = DecisionTreeRegressor(splitter='best', max_depth=13, max_features=30, random_state=0)
tree.fit(X_train, y_train)


# Evaluating Decision Tree
y_pred_tree = rfr.predict(x_test)
print("The regression score of Decision Tree is", 100 * tree.score(x_test, y_test), "%")
print("The MSE of Decision Tree is", mean_squared_error(y_test, y_pred_tree))
print("The RMSE of Decision Tree is", np.sqrt(mean_squared_error(y_test, y_pred_tree)))
print("The MAE of Decision Tree is", mean_absolute_error(y_test, y_pred_tree))

kfoldscore_tree = cross_val_score(tree, X, y, cv=kfold, scoring='r2').mean()
print("The KFold score of Decision Tree is", 100 * kfoldscore_tree, "%")

model_metrics = {
    'RFR Score': [rfr.score(x_test, y_test)],
    'RFR MSE': [mean_squared_error(y_test, y_pred)],
    'RFR RMSE': [np.sqrt(mean_squared_error(y_test, y_pred))],
    'RFR MAE': [mean_absolute_error(y_test, y_pred)],
    'RFR OOB Score': [rfr.oob_score_],
    'RFR KFold Score': [kfoldscore_rfr],
    'Decision Tree Score': [tree.score(x_test, y_test)],
    'Decision Tree MSE': [mean_squared_error(y_test, y_pred_tree)],
    'Decision Tree RMSE': [np.sqrt(mean_squared_error(y_test, y_pred_tree))],
    'Decision Tree MAE': [mean_absolute_error(y_test, y_pred_tree)],
    'Decision Tree KFold Score': [kfoldscore_tree]
}

metrics_df = pd.DataFrame(model_metrics)
print(metrics_df)



# Hyperparameter tuning

model_params = {
    'random_forest': {
        'model': RandomForestRegressor(random_state=0),
        'params': {
            'n_estimators': [95, 96, 97, 98, 99, 100],
            'max_depth': [13, 14, 15, 16],
            'max_features': [31, 33, 35, 37]
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor(random_state=0),
        'params': {
            'splitter': ['best'],
            'max_depth': [13, 14, 15, 16, 17],
            'max_features': [30, 31, 32, 33, 34, 35, 36]
        }
    }
}

scores = []
for model_name, model_parameters in model_params.items():
    clf = GridSearchCV(model_parameters['model'], model_parameters['params'], cv=10, return_train_score=False)
    clf.fit(X_train, y_train)
    scores.append({
        'model': model_name,
        'best score': clf.best_score_,
        'best params': clf.best_params_
    })
    print(clf.best_params_)
scores_df = pd.DataFrame(scores)
print(scores_df)

