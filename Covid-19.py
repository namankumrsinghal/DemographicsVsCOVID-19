import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

cwd = os.getcwd()

print(cwd)

data = pd.read_csv(cwd + "\\COVID-19_Case_Surveillance_Public_Use_Data.csv")

#print("Total number of observations ", len(data))
#number of patients who were assymptomatic
#print("No. of assymptomatic patients", data["onset_dt"].isnull().sum())

#slicing the dataset to have only observations with symptoms
data_symptom = data[data["onset_dt"].notnull()]
#print("Number of symptomatic patients ", len(data_symptom))

#let's take a random sample of 6000000 observations for our project
symptom = data_symptom.sample(n = 6000000)

#slicing the dataset to have only observations without symptoms
data_assymptom = data[data["onset_dt"].isna()]

#CLEANING THE DATA-----------------------------------------------------------------------------------------------------------------------
#let's clean up the dataset(only use the features that are important to our model), and remove NA and missing values
symptom = symptom[["onset_dt", "age_group", "sex", "race_ethnicity_combined"]]
symptom = symptom[symptom["age_group"].notnull()]
symptom = symptom[symptom["age_group"] != "Missing"]

#for this assignment, we only consider male and female gender observations
symptom = symptom[symptom["sex"].notnull()]
symptom = symptom[symptom["sex"] != "Missing"]
symptom = symptom[symptom["sex"] != "Unknown"]
symptom = symptom[symptom["sex"] != "Other"]

symptom = symptom[symptom["race_ethnicity_combined"].notnull()]
symptom = symptom[symptom["race_ethnicity_combined"] != "Missing"]
symptom = symptom[symptom["race_ethnicity_combined"] != "Unknown"]

race = list()
ethnicity = list()

#separating race and ethnicity
for i in range(len(symptom)):
    r_e = symptom["race_ethnicity_combined"].iloc[i]
    if r_e.find(',') == -1:     #no comma found - this implies that at least one of race or ethnicity is missing
        race.append("Missing")
        ethnicity.append("Missing")
    else:
        race.append(r_e[ : r_e.find(',')])
        ethnicity.append(r_e[r_e.find(',') + 2 : ])
symptom["race"] = race
symptom["ethnicity"] = ethnicity
symptom = symptom[symptom["race"] != "Missing"]

#drop the column race_ethnicity_combined, onset_dt and add a column "Symptoms" with values "Yes"
symptom_list = ["Yes"]*len(symptom)
symptom = symptom[["age_group", "sex", "race", "ethnicity"]]
symptom["symptom"] = symptom_list

#print(len(symptom))
#the data set is now clean

#let's consider 1000000 random observations from the above cleaned dataframe to build and test our models
symptom = symptom.sample(n = 1000000)
#print(symptom.head())
#print(symptom["race"])
#print(symptom["ethnicity"])

#print("Total number of observations with symptoms", len(data_symptom))
#print(data_symptom.head())



#let's take a random sample of 6000000 observations from asymptomatic dataset for our project
assymptom = data_assymptom.sample(n = 6000000)
#let's clean up the dataset(only use the features that are important to our model), and remove NA and missing values
assymptom = assymptom[["onset_dt", "age_group", "sex", "race_ethnicity_combined"]]
assymptom = assymptom[assymptom["age_group"].notnull()]
assymptom = assymptom[assymptom["age_group"] != "Missing"]

#for this assignment, we only consider male and female gender observations
assymptom = assymptom[assymptom["sex"].notnull()]
assymptom = assymptom[assymptom["sex"] != "Missing"]
assymptom = assymptom[assymptom["sex"] != "Unknown"]
assymptom = assymptom[assymptom["sex"] != "Other"]

assymptom = assymptom[assymptom["race_ethnicity_combined"].notnull()]
assymptom = assymptom[assymptom["race_ethnicity_combined"] != "Missing"]
assymptom = assymptom[assymptom["race_ethnicity_combined"] != "Unknown"]

race = list()
ethnicity = list()

#separating race and ethnicity
for i in range(len(assymptom)):
    r_e = assymptom["race_ethnicity_combined"].iloc[i]
    if r_e.find(',') == -1:     #no comma found - this implies that at least one of race or ethnicity is missing
        race.append("Missing")
        ethnicity.append("Missing")
    else:
        race.append(r_e[ : r_e.find(',')])
        ethnicity.append(r_e[r_e.find(',') + 2 : ])
assymptom["race"] = race
assymptom["ethnicity"] = ethnicity
assymptom = assymptom[assymptom["race"] != "Missing"]

#drop the column race_ethnicity_combined, onset_dt and add a column "Symptoms" with values "No"
assymptom_list = ["No"]*len(assymptom)
assymptom = assymptom[["age_group", "sex", "race", "ethnicity"]]
assymptom["symptom"] = assymptom_list

#the data set is now clean

#print(len(assymptom))
#let's consider 1000000 random observations from the above cleaned dataframe to build and test our models
assymptom = assymptom.sample(n = 1000000)
#print(assymptom.head())

#print("Total number of observations with no symptoms", len(data_assymptom))
#print(data_assymptom.head())

#join the two datasets
#can think of manually setting onset_dt = NaN as symptomatic = no in assymptom and as YES in symptom
data = symptom.append(assymptom)

#END OF CLEANING THE DATA----------------------------------------------------------------------------------------------------------------

def roc_plotter(Y_true, Y_pred):
    """plots the roc curve(c statistic)"""
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = metrics.roc_curve(Y_true, Y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    #ROC curve
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')



#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data[["age_group", "sex", "race", "ethnicity"]]

dummies = [pd.get_dummies(data[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression

log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

roc_plotter(Y_test, Y_pred_logistic)
plt.title('Receiver Operating Characteristic for Logistic Regression')
plt.show()

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

roc_plotter(Y_test, Y_pred_decisionTree)
plt.title('Receiver Operating Characteristic for Decision tree classifier')
plt.show()

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

roc_plotter(Y_test, Y_pred_randomForest)
plt.title('Receiver Operating Characteristic for Random Forest classifier')
plt.show()

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

roc_plotter(Y_test, Y_pred_LDA)
plt.title('Receiver Operating Characteristic for LDA classifier')
plt.show()

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

roc_plotter(Y_test, Y_pred_NB)
plt.title('Receiver Operating Characteristic for Naive bayesian classifier')
plt.show()

#Removing Age group as a feature------------------------------------------------------------------------------------
print("Removing age from the feature set")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data[["sex", "race", "ethnicity"]]

dummies = [pd.get_dummies(data[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

#Removing sex as a feature
print("Removing sex from the feature set")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data[["age_group", "race", "ethnicity"]]

dummies = [pd.get_dummies(data[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

#Removing race as a feature
print("Removed race from the feature set")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data[["age_group", "sex", "ethnicity"]]

dummies = [pd.get_dummies(data[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

#Removing ethnicity as a feature
print("Removing ethnicity from the feature set")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data[["age_group", "sex", "race"]]

dummies = [pd.get_dummies(data[c]) for c in input_data.columns]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

"""
#TAKING FEAUTRES ONE AT A TIME TO BUILD SIMPLE MODELS(uncomment this code block to see the results)
#Feature = Age------------------------------------------------------------------------------------
print("\nAge\n")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data["age_group"]

dummies = [pd.get_dummies(data["age_group"])]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

#Feature = Sex
print("\nGender\n")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data["sex"]

dummies = [pd.get_dummies(data["sex"])]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

#Feature = race
print("\nRace\n")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data["race"]

dummies = [pd.get_dummies(data["race"])]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda_classifier = LDA()
lda_classifier.fit(X_train, Y_train)

Y_pred_LDA = lda_classifier.predict(X_test)

accuracy_LDA = lda_classifier.score(X_test, Y_test)

print("Accuracy with LDA classifier is", accuracy_LDA)

#Naive Bayesian
from sklearn.naive_bayes import BernoulliNB

NB_Classifier = BernoulliNB().fit(X_train, Y_train)
Y_pred_NB = NB_Classifier.predict(X_test)

accuracy_NB = np.mean(Y_pred_NB == Y_test)

print("Accuracy with Naive Bayesian is ", accuracy_NB)

#Feature = ethnicity
print("\nethnicity\n")

#input data(creating dummy variables for categorical variables and splitting into test and train) for all the classification models:
input_data = data["ethnicity"]

dummies = [pd.get_dummies(data["ethnicity"])]
binary_data = pd.concat(dummies, axis = 1)
X = binary_data[0:len(data)].values
le = LabelEncoder()
Y = le.fit_transform(data["symptom"].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.19, random_state = 19)

#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression


log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(X_train, Y_train)

Y_pred_logistic = log_reg_classifier.predict(X_test)
accuracy_logistic = np.mean(Y_pred_logistic == Y_test)
print("Accuracy with logistic regression is", accuracy_logistic)

#DECISION TREE
from sklearn import tree

clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X_train, Y_train)

Y_pred_decisionTree = clf.predict(X_test)

accuracy_decisionTree = np.mean(Y_pred_decisionTree == Y_test)
print("Accuracy with decision tree is", accuracy_decisionTree)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5, max_depth=5, criterion="entropy")
model.fit(X_train, Y_train)
Y_pred_randomForest = model.predict(X_test)

accuracy_randomForest = np.mean(Y_pred_randomForest == Y_test)
print("Accuracy with random forest is", accuracy_randomForest)
"""