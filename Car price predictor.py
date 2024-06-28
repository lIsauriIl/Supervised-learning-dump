import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import f_oneway, chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# DATA LOADING
df = pd.read_csv(r"c:\Users\user\Downloads\cardekho.csv")
df.columns = ['Car name', 'Year', 'Selling price', 'Km driven', 'Fuel type', 'Seller type', 'Transmission',
              'Owner', 'Mileage', 'Engine', 'Max power', 'Seats']




# INITIAL PREPROCESSING

# Removing rows with NaN values and duplicate rows
df = df.dropna()
df = df.drop_duplicates()

# EDA AND VISUALISATION

# Describing the dataset features' mean, count, stddev, min, no. of rows & columns of df
#print(df.describe())
#print(df.shape)

# Verifying there are no null entries after initial cleaning
#print(df.isna().sum())
'''
# Plotting the counts of the discrete variables
sns.countplot(df, x='Fuel type', hue='Fuel type')
plt.show()
sns.countplot(df, x='Transmission', hue='Transmission')
plt.show()
sns.countplot(df, x='Owner', hue='Owner')
plt.show()
sns.countplot(df,x='Seller type', hue='Seller type')
plt.show()
sns.countplot(df, x='Seats', hue='Seats')
plt.show()'''

'''ANALYSIS OF COUNTS: None of the fuel types seem insignificant enough to exclude.
For Owner, although test drive is extremely rare, it is responsible for some of the 
highest prices on the dataset, so it's a good idea to not exclude it, instead
finding a way to handle it. For seller type, trustmark dealer is relatively low compared 
to the other entries, but it might hold some value based on future EDA.
Both 2 and 14 on the seat counts are extremely low. However, 2 is responsible for veryhigh
selling prices, but 14 remains to be seen.'''


# Plotting correlation heat map

df_heatmap = df[['Year', 'Selling price', 'Km driven', 'Mileage', 'Engine', 'Max power']]
df_heatmap['Max power'] = pd.to_numeric(df_heatmap['Max power'], errors='coerce')
df['Max power'] = pd.to_numeric(df['Max power'], errors='coerce')
#print(df_heatmap['Max power'].unique())
#sns.heatmap(df_heatmap.corr(), annot=True, cmap='RdBu')
#plt.show()
'''HEATMAP ANALYSIS: Selling price has the strongest correlation with max power
at 0.69. Selling price has decent and near equal correlation with year and engine (0.43/44), however,
those 2 variables aren't correlated. Instead, engine and max power are decently correlated,
so I might consider removing engine, or using lasso regression. Selling price has a slight 
negative correlation with both km driven and mileage, though the 2 don't affect each other much either.
We will remove engine to save time on testing alpha hyperparameter. Also remove Mileage and KM driven
because its correlation with price is extremely weak'''


# Plotting box plots with categorical data against price
'''sns.boxplot(x = "Fuel type", y = "Selling price", showmeans=True, data=df, showfliers=True)
plt.show()

sns.boxplot(x = "Owner", y = "Selling price", showmeans=True, data=df, showfliers=True)
plt.show()

sns.boxplot(x = "Seats", y = "Selling price", showmeans=True, data=df, showfliers=True)
plt.show()

sns.boxplot(x = "Transmission", y = "Selling price", showmeans=True, data=df, showfliers=True)
plt.show()

sns.boxplot(x = "Seller type", y = "Selling price", showmeans=True, data=df, showfliers=True)
plt.show()'''

'''REMOVING SEATS AS THE MOST COMMON DATATYPE BY 5 IS 5, AND IS RELATIVELY EVENLY DISTRIBUTED.
OTHER SEAT NUMBER SOMEWHAT COMPARABLE IN INFLUENCE IS 7.
OUT OF ALL THE CATEGORIES, TRANSMISSION SEEMS TO YIELD THE MOST OBVIOUS RESULT IN TERMS OF THE 
INFLUENCE ON PRICE, WITH AUTOMATIC ON AVERAGE BEING HIGHER PRICED.
FOR SELLER TYPE, DEALER ON AVERAGE IS PRICED HIGHER THAN INDIVIDUAL, SHOWING SOME SORT OF CORRELATION.'''


# Chi tests (to save on space and code, I've done the testing by deleting and renaming columns)
contingency_table = pd.crosstab(df['Seller type'], df['Transmission'])
#print(contingency_table)
chi2, pval, degrees_of_freedom, expected_table = chi2_contingency(contingency_table)
print(pval)


'''ANALYSIS: Fuel type and owner have a statistically significant positive correlation.
Fuel type and transmission have an even stronger positive correlation so we can remove fuel type.
Owner and transmission have no correlation at all, with the pvalue being almost 5.
Seller type and transmission have an incredibly strong correlation, as most individuals sell
manual transmission cars. So we will keep only owner and transmission'''


'''END RESULT: We will remove engine, mileage, km driven, seats, fuel type, seller type and name.
UPDATE: THE ABOVE CHANGES IS ONLY FOR LINEAR REGRESSION, NOT RANDOM FOREST'''

# POST EDA PROCESSING

#df = df.drop('Engine', axis=1).drop('Mileage', axis=1).drop('Km driven', axis=1).drop('Seats', axis=1)
#df = df.drop('Fuel type', axis=1).drop('Seller type', axis=1).drop('Car name', axis=1)
#print(df.head()) # To verify that we've dropped correctly
df = df.drop('Car name', axis=1).drop('Seats', axis=1)

# Label encoding
transmission_df = pd.get_dummies(df['Transmission'], drop_first=True)
owner_df = pd.get_dummies(df['Owner'], drop_first=True)
fuel_df = pd.get_dummies(df['Fuel type'], drop_first=True)
seller_df = pd.get_dummies(df['Seller type'], drop_first=True)
sns.lmplot(df, x='Max power', y='Selling price', hue='Fuel type')
#plt.show()

df = pd.concat([transmission_df, df], axis=1)
df = pd.concat([owner_df, df], axis=1)
df = pd.concat([seller_df, df], axis=1)
df = pd.concat([fuel_df, df], axis=1)
df.drop('Transmission', axis=1, inplace=True)
df.drop('Owner', axis=1, inplace=True)
df.drop('Fuel type', axis=1, inplace=True)
df.drop('Seller type', axis=1, inplace=True)
#print(df.head())

# Generating a sample for training and storing original
og_df = df
df = df.sample(500)

# TRAINING

# Splitting data
X = df.drop('Selling price', axis=1)
y = df['Selling price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Initializing random forest regression
model = RandomForestRegressor(n_estimators=464, max_depth=20, oob_score=True)
model.fit(X_train, y_train)
print("The regression score is", 100 * model.score(X_test, y_test), "%")


# Metrics
 
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("The MSE is", mean_squared_error(y_test, y_pred))
print("The MAE is", mean_absolute_error(y_test, y_pred))
print("The RMSE is", np.sqrt(mean_squared_error(y_test, y_pred)))
print("The model's OOB error is", model.oob_score_)
# Note: Sometimes, the prediction model doesn't work. It has something to do with inability to convert from string to float

# GRIDSEARCHCV



# METRIC DATAFRAME
metrics = {
    "R2 Score" : [model.score(X_test, y_test)],
    "RMSE" : [rmse],
    "MSE" : [mean_squared_error(y_test, y_pred)],
    "MAE" : [mean_absolute_error(y_test, y_pred)]
}

metric_df = pd.DataFrame(metrics)
print(metric_df)