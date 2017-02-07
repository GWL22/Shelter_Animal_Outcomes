# coding=utf-8

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./P05_shelter_animal/dataset/train_labeling.csv')
temp = pd.read_csv('./P05_shelter_animal/dataset/pre_test.csv')
del temp['DateTime']

feature_names = ['AnimalType',
                 'SexuponOutcome',
                 # 'AgeuponOutcome',
                 'AgeOutcomeMonth',
                 # 'AgeOutcomeYear',
                 # 'IntakeYear',
                 # 'IntakeMonth',
                 # 'IntakeDay',
                 # 'IntakeTime',
                 'Color',
                 'Breed'
                 ]
dfX = df[feature_names]
dfy = df['OutcomeType']
test = temp[feature_names]

# dataset
# ds1 => train1.csv
# ds2 => train_labeling.csv
# ds3 => train_20170202.csv

print temp.tail()
# X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.25)

# ds1 max_depth = 5, 0.61, 0.62, 0.55
# ds2 max_depth = 5, 0.64, 0.63, 0.61
model = DecisionTreeClassifier(max_depth=5)

# ds2 0.60 0.59 0.55
# model2 = SVC(kernel='linear', probability=True)

# ds1 0.59, 0.61, 0.59 가우시안 사용시
# ds2 0.58 0.59 0.58 가우시안 사용시
# model2 = OneVsOneClassifier(GaussianNB())

# ds1 0.58, 0.60, 0.57 가우시안 사용시
# ds2 0.58 0.60 0.58 가우시안 사용시
# model3 = OneVsRestClassifier(GaussianNB())

# ds1 0.58, 0.60, 0.57
# ds2 0.59, 0.60, 0.59
# model = GaussianNB()

# ds1 0.60 0.60 0.58
# ds2 0.60 0.62 0.60
# qda = QuadraticDiscriminantAnalysis()

# ds2 0.57 0.58 0.55
# model = LogisticRegression()

# model4 = VotingClassifier(estimators=[('DT', model1), ('GNB', model2)],
#                           voting='soft', weights=[1, 2])
# model5 = BaggingClassifier(model1, bootstrap_features=True)
# model6 = RandomForestClassifier(max_depth=5, n_estimators=5)
# model7 = ExtraTreesClassifier(max_depth=5, n_estimators=30)
setting = model.fit(dfX, dfy)
# print classification_report(y_test, setting.predict(X_test))

result = setting.predict(test)
result_proba = setting.predict_proba(test)
result_proba_pd = pd.DataFrame(data=result_proba, columns=['Adoption', 'Died', 'Euthanasia', 'Return_to_owner', 'Transfer'])
predict_result = pd.concat([temp['ID'], result_proba_pd], axis=1)

predict_result.to_csv('Shelter Animal Outcomes.csv', index=False)
print 'complete'
