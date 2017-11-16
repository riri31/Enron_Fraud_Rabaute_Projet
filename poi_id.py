#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from poi_id_functions import data_cleaning, Draw, Draw_bar, feature_NaN, feature_NaN_poi, feature_analysis, Remove_NaN_Person, Count_NaN, List_NaN, Add_ratio, print_rank, tester, classifier_test

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.cross_validation import train_test_split

from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV


############ Code generate verbose print or not ############
verbose = False
############################################################


######### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

######### Feature selection
print '\n########### STEP 1: Feature selection ############\n'
    ## Selected features
features_list = ['poi','total_payments','total_stock_value','exercised_stock_options', 'from_poi_to_this_person','shared_receipt_with_poi','from_this_person_to_poi']
    ## Removed features
removed_features_list=['loan_advances','deferred_income','long_term_incentive','deferral_payments','restricted_stock_deferred','director_fees']
#features_list+=removed_features_list

                        
    ## How many persons:   
print '\n## Number of persons on the sample: {}'.format(len(data_dict))
    
    ## How many poi:
poi_count=0
for i in data_dict: poi_count+=data_dict[i]['poi']
print '## Number of poi on the sample: {} ({}%)'.format(poi_count,round((1.0*poi_count)/len(data_dict),2))

    ## print count for NaN features
if verbose: print_rank(feature_NaN(data_dict),'\nNumber of NaN value for each feature:',True)

    ## Correlation between NaN and POI status
if verbose: Draw_bar(feature_NaN_poi(data_dict, NaN=True),title='POI status with NaN feature')
if verbose: Draw_bar(feature_NaN_poi(data_dict, NaN=False),title='POI status with no NaN feature')

    ## print ratio between poi and not poi for NaN values, order by ratio value descending
dict_non_nan=feature_NaN_poi(data_dict, NaN=False)
dict_ratio={}
for i in dict_non_nan:
    poi=dict_non_nan[i]['poi']
    not_poi=dict_non_nan[i]['not_poi']
    dict_ratio[i]=round((poi*1.0)/(poi+not_poi),2)

j=1
if verbose: print '\nratio between poi and not poi for features different from NaN:'
for k, v in sorted(dict_ratio.items(), key=itemgetter(1), reverse=True):
    if verbose: print '({}) {}:\tpoi={}% (n={})'.format(j,k,v,dict_non_nan[k]['poi']+dict_non_nan[k]['not_poi'])
    j+=1

######### Remove outliers

print '\n\n########### STEP 2: Remove outliers ############\n'

    ## print Number of persons before cleaning
print '## Number of persons before cleaning: {}\n'.format(len(data_dict))

    ## print max value for each features
for i in data_dict.values()[0]:
    if verbose: print 'Maximum value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='max'))
    ## print min value for each features
for i in data_dict.values()[0]:
    if verbose: print 'Min value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='min'))

    ## Remove TOTAL from list of person as error
data_cleaning(data_dict,['TOTAL'])

    ## Remove person with NaN in all selected features
Remove_NaN_Person(data_dict,features_list[1::],number_nan=len(features_list[1::]))

    ## print Number of persons after cleaning
print '\n## Number of persons after cleaning: {}\n'.format(len(data_dict))

    ## print max value for each features after the cleaning
for i in data_dict.values()[0]:
    if verbose: print 'Maximum value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='max'))

    ## print min value for each features after the cleaning
for i in data_dict.values()[0]:
    if verbose: print 'Min value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='min'))

    ## print number and list of NaN per person of the sam^ple
dict_nan=Count_NaN(data_dict, features_list[1::])
if verbose: print '\nNumber of NaN per person:'
j=1
for k, v in sorted(dict_nan.items(), key=itemgetter(1), reverse=True):
    if verbose: print '({}) {}:\tNAN={}/{}:{}'.format(j,k,v,len(features_list[1::]),List_NaN(data_dict[k], features_list))
    j+=1

######### New features

print '\n\n########### STEP 3: Add new features ############\n'
my_dataset = dict(data_dict)

print '\n## Number of features before new features: {}\n'.format(len(features_list))
    ## add ratio exercised stock options / total payments
my_dataset , features_list = Add_ratio(features_list=features_list,data_dict=data_dict,new_feature_name='ratio_exer_stock_total',feature_1='exercised_stock_options',feature_2='total_payments',operator='/')
    ## add ratio from poi to this person / total email received
my_dataset , features_list = Add_ratio(features_list=features_list,data_dict=data_dict,new_feature_name='ratio_from_poi_email_received',feature_1='from_poi_to_this_person',feature_2='to_messages',operator='/')
    ## add ratio from this person to poi/ total email sent
my_dataset , features_list = Add_ratio(features_list=features_list,data_dict=data_dict,new_feature_name='ratio_to_poi_email_sent',feature_1='from_this_person_to_poi',feature_2='from_messages',operator='/')
    ## add ratio exercised stock options / total stock
#my_dataset , features_list = Add_ratio(features_list=features_list,data_dict=data_dict,new_feature_name='ratio_exer_stock_total_stock',feature_1='exercised_stock_options',feature_2='total_stock_value',operator='/')

print '\n## Number of features after new features: {}\n'.format(len(features_list))

    ## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

    ## scale all features
scaler = MinMaxScaler()
features=scaler.fit_transform(features)

    ## Select best features
k='all'

selector= SelectKBest(chi2, k=k)
new_features=selector.fit_transform(features, labels)
if k!='all':
    new_features_list=['poi']
    for i in selector.get_support(indices=True):
        new_features_list.append(features_list[i+1])
    features_list=new_features_list
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    
    
Scores_features={}
j=0
for i in features_list[1::]:
    Scores_features[i]=selector.scores_[j]
    j+=1

print '\n\n########### STEP 4: Score and rank features ############\n'
print_rank(Scores_features,'\nScores for features:\n',True)

    ##Draw scatter plot 2 by 2 with score descending
print '\n\n########### STEP 5: Draw scatter plots ############\n'
features_order=sorted(Scores_features.items(), key=itemgetter(1), reverse=True)
j=0
while j<len(Scores_features)-1:
    if verbose: Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_"+str(j)+".png", f1_name=features_order[j][0], f2_name=features_order[j+1][0])
    j+=2
    ##Draw one specific scatter

if verbose: Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_"+str(j)+".png", f1_name='exercised_stock_options', f2_name='ratio_from_poi_email_received')
if verbose: Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_"+str(j+1)+".png", f1_name='ratio_to_poi_email_sent', f2_name='ratio_from_poi_email_received')

######### Try different classifiers
print '\n\n########### STEP 6: Try different classifier (default params) ############\n'
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

    ##KNeighborsClassifier
clf=KNeighborsClassifier()
classifier_test(clf, my_dataset, features_list,'KNeighborsClassifier (default):')

    ##Decision Tree
clf=DecisionTreeClassifier(random_state=5)
classifier_test(clf, my_dataset, features_list,'Decision tree (default):')

    ##Decision Tree
clf=LinearSVC()
classifier_test(clf, my_dataset, features_list,'Linear SVC (default):')


######### Tune the algorithm


#### Tune Decision Tree
print '\n\n########### STEP 7: Tune (with grid search) ############\n'

                                    
scoring='precision'
    ### set the parameters for decision tree classifer
criterion=['gini','entropy']
max_depth=np.arange(1,100,5)
min_samples_split=np.arange(2,20,1)
random_state=[5]
params_decision_tree = dict(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split,random_state=random_state)

    ### set the classifier
classifier=DecisionTreeClassifier()
    ### fit and search
estimator = GridSearchCV(classifier, params_decision_tree, scoring=scoring, cv=None)
estimator.fit(features_train, labels_train)
    ### extract scores
score_k_best = estimator.cv_results_
    ### get the best estimator
clf = estimator.best_estimator_
    ### Test the best estimator
classifier_test(clf, my_dataset,features_list,'Grid search (Decision Tree):')

#### Tune Decision Tree

    ### set the parameters for kbest classifer
metrics= ['minkowski','euclidean','manhattan'] 
weights= ['uniform','distance']
numNeighbors= np.arange(3,12,1)

params_k_best = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
    ### set the classifier
classifier=KNeighborsClassifier()
    ### fit and search
estimator = GridSearchCV(classifier, params_k_best, scoring=scoring, cv=None)
estimator.fit(features_train, labels_train)
    ### extract scores
score_k_best = estimator.cv_results_
    ### get the best estimator
clf = estimator.best_estimator_
    ### Test the best estimator
classifier_test(clf, my_dataset,features_list,'Grid search (K best):')


######### Dump results
print '\n\n########### STEP 8: Dump results ############\n'
dump_classifier_and_data(clf, my_dataset, features_list)
print 'done.\n'