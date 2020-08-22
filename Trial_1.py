from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from sklearn.impute import SimpleImputer

raw = 'https://raw.githubusercontent.com/owid/covid-19-data/2a9c16550789163b5d1bfeff780d33b9d7713d1f/public/data/owid-covid-data.csv'
urlretrieve(raw)
df = pd.read_csv(raw, sep=',')
attributes = ["iso_code","continent","location","date","total_cases","new_cases","total_deaths","new_deaths","total_cases_per_million","new_cases_per_million","total_deaths_per_million","new_deaths_per_million","new_tests","total_tests","total_tests_per_thousand","new_tests_per_thousand","new_tests_smoothed","new_tests_smoothed_per_thousand","tests_per_case","positive_rate","tests_units","stringency_index","population","population_density","median_age","aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","cardiovasc_death_rate","diabetes_prevalence","female_smokers","male_smokers","handwashing_facilities","hospital_beds_per_thousand","life_expectancy"]
num_attributes = ["iso_code","continent","location","date","total_cases","new_cases","total_deaths","new_deaths","total_cases_per_million","new_cases_per_million","total_deaths_per_million","new_deaths_per_million","new_tests","total_tests","total_tests_per_thousand","new_tests_per_thousand","new_tests_smoothed","new_tests_smoothed_per_thousand","tests_per_case","positive_rate","tests_units","stringency_index","population","population_density","median_age","aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","cardiovasc_death_rate","diabetes_prevalence","female_smokers","male_smokers","handwashing_facilities","hospital_beds_per_thousand","life_expectancy"]
df.columns = attributes

nd = df

nd["date"] = nd["date"].replace('-', '', regex=True).astype(int)

nd["total_cases"].fillna(value=nd["total_cases"].mean(), inplace=True)
nd["total_tests"].fillna(value=nd["total_cases"].mean(), inplace=True)
nd["new_cases"].fillna(value=nd["new_cases"].mean(), inplace=True)
nd["population"].fillna(value=nd["population"].mean(), inplace=True)
nd["total_deaths"].fillna(value=nd["total_deaths"].mean(), inplace=True)
nd["new_deaths"].fillna(value=nd["new_deaths"].mean(), inplace=True)
nd["total_cases_per_million"].fillna(value=nd["total_cases_per_million"].mean(), inplace=True)
nd["new_tests"].fillna(value=nd["new_tests"].mean(), inplace=True)
nd["new_cases_per_million"].fillna(value=nd["new_cases_per_million"].mean(), inplace=True)
nd["total_deaths_per_million"].fillna(value=nd["total_deaths_per_million"].mean(), inplace=True)
nd["new_deaths_per_million"].fillna(value=nd["new_deaths_per_million"].mean(), inplace=True)
nd["total_tests"].fillna(value=nd["total_tests"].mean(), inplace=True)
nd["total_tests_per_thousand"].fillna(value=nd["total_tests_per_thousand"].mean(), inplace=True)
nd["new_tests_per_thousand"].fillna(value=nd["new_tests_per_thousand"].mean(), inplace=True)
nd["stringency_index"].fillna(value=nd["stringency_index"].mean(), inplace=True)
nd["population_density"].fillna(value=nd["population_density"].mean(), inplace=True)
nd["median_age"].fillna(value=nd["median_age"].mean(), inplace=True)
nd["aged_65_older"].fillna(value=nd["aged_65_older"].mean(), inplace=True)
nd["aged_70_older"].fillna(value=nd["aged_70_older"].mean(), inplace=True)
nd["gdp_per_capita"].fillna(value=nd["gdp_per_capita"].mean(), inplace=True)
nd["extreme_poverty"].fillna(value=nd["extreme_poverty"].mean(), inplace=True)
nd["cardiovasc_death_rate"].fillna(value=nd["cardiovasc_death_rate"].mean(), inplace=True)
nd["diabetes_prevalence"].fillna(value=nd["diabetes_prevalence"].mean(), inplace=True)
nd["female_smokers"].fillna(value=nd["female_smokers"].mean(), inplace=True)
nd["male_smokers"].fillna(value=nd["male_smokers"].mean(), inplace=True)
nd["life_expectancy"].fillna(value=nd["life_expectancy"].mean(), inplace=True)

X = nd[["date","total_cases","new_cases","total_tests_per_thousand","new_tests_per_thousand","stringency_index","population","population_density","median_age","aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","diabetes_prevalence","cardiovasc_death_rate","female_smokers","male_smokers","life_expectancy"]].values
Y = nd[["total_deaths"]].values
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y.ravel(), test_size=0.5, random_state=101)


rf = RandomForestRegressor()
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)

print(rf.feature_importances_ )
print(mean_absolute_error(Y_test, Y_pred))
features = rf.feature_importances_.tolist()
features.pop(17)
features.pop(12)
features.pop(11)
features.pop(9)
features.pop(6)
features.pop(5)
features.pop(4)
features.pop(2)
features.pop(1)

names =  ["date","total_tests","population_density","median_age","aged_70_older","diabetes_prevalence", "cardiovasc_death_rate","female_smokers","male_smokers"]
names =  ["date", "total_cases","new_cases","total_tests","new_tests","stringency_index","population","population_density","median_age","aged_65_older","aged_70_older","gdp_per_capita","extreme_poverty","diabetes_prevalence", "cardiovasc_death_rate","female_smokers","male_smokers","life_expectancy"]
font = {'family' : 'normal',
        'size'   : 6}
plt.rc('font', **font)
plt.bar(names,features, width=0.3)
plt.show()


Xtg1 = pd.DataFrame(X_test)
Xtg2 = pd.DataFrame(X_test)
Ytg = pd.DataFrame(Y_test)
Ypg = pd.DataFrame(Y_pred)
Xtg1["results"] = Ypg[0]
Xtg1["date"] = Xtg1[0]
Xtg2["actual"] = Ytg[0]
Xtg2["date"] = Xtg2[0]
sns.set_palette("muted")
sns.scatterplot(x="date", y="actual", data = Xtg2) #now this is blue ; this is our actual results
sns.scatterplot(x="date", y="results", data = Xtg1) #now this is orange; this is the predictions
plt.show()



