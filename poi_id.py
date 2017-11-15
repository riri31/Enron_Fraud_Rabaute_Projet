#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import poi_id_functions



##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### baseline features_list = ['poi','salary'] # You will need to use more features
#features_list = ['poi','salary', 'total_stock_value']
features_list = ['poi','exercised_stock_options','total_stock_value', 'total_payments','from_poi_to_this_person','shared_receipt_with_poi','from_this_person_to_poi']
removed_features_list=['loan_advances','deferred_income','long_term_incentive','deferral_payments','restricted_stock_deferred','director_fees']
#features_list+=removed_features_list

                             
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##

## First observations
## How many persons:

print '\n########### STEP 1: Feature selection ############\n'
    
total_count=len(data_dict)
print '\nNumber of persons on the sample: {}'.format(total_count)
    
## How many poi:
poi_count=0
for i in data_dict:
    poi_count+=data_dict[i]['poi']
    
print '\nNumber of poi on the sample: {} ({}%)'.format(poi_count,round((1.0*poi_count)/total_count,2))

## print count for NaN features
from operator import itemgetter
j=1
print '\nNumber of NaN value for each feature:'
for k, v in sorted(feature_NaN(data_dict).items(), key=itemgetter(1), reverse=True):
    print '({}) {}:{}'.format(j,k,v)
    j+=1



## Correlation between NaN and POI status


Draw_bar(feature_NaN_poi(data_dict, NaN=True),title='POI status with NaN feature')
Draw_bar(feature_NaN_poi(data_dict, NaN=False),title='POI status with no NaN feature')


## print ratio between poi and not poi for NaN values, order by ratio value descendinf
dict_non_nan=feature_NaN_poi(data_dict, NaN=False)

dict_ratio={}
for i in dict_non_nan:
    poi=dict_non_nan[i]['poi']
    not_poi=dict_non_nan[i]['not_poi']
    dict_ratio[i]=round((poi*1.0)/(poi+not_poi),2)

j=1
print '\nratio between poi and not poi for features different from NaN:'
for k, v in sorted(dict_ratio.items(), key=itemgetter(1), reverse=True):
    print '({}) {}:\tpoi={}% (n={})'.format(j,k,v,dict_non_nan[k]['poi']+dict_non_nan[k]['not_poi'])
    j+=1

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Task 2: Remove outliers

##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##

print '\n\n########### STEP 2: Remove outlies ############\n'
print '## Number of persons before cleaning: {}\n'.format(len(data_dict))

for i in data_dict.values()[0]:
    print 'Maximum value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='max'))

for i in data_dict.values()[0]:
    print 'Min value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='min'))

## I remove TOTAL from list of person as error
data_cleaning(data_dict,['TOTAL'])

## I remove 'BHATNAGAR SANJAY','DERRICK JR. JAMES V','BELFER ROBERT','RICE KENNETH D'  from list of person as negative value -> cancelled as not effective
#data_cleaning(data_dict,['BHATNAGAR SANJAY','DERRICK JR. JAMES V','BELFER ROBERT','RICE KENNETH D'])

## I remove person with NaN in all selected features
Remove_NaN_Person(data_dict,features_list[1::],number_nan=len(features_list[1::]))

print '\n## Number of persons after cleaning: {}\n'.format(len(data_dict))


for i in data_dict.values()[0]:
    print 'Maximum value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='max'))

for i in data_dict.values()[0]:
    print 'Min value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='min'))


print '\nFeatures for max Salary:{}'.format(data_dict[feature_analysis(data_dict,'salary',analysis='max')[0]])


dict_nan=Count_NaN(data_dict, features_list[1::])

print '\nNumber of NaN per person:'
j=1
for k, v in sorted(dict_nan.items(), key=itemgetter(1), reverse=True):
    print '({}) {}:\tNAN={}/{}:{}'.format(j,k,v,len(features_list[1::]),List_NaN(data_dict[k], features_list))
    j+=1

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


my_dataset = data_dict

##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##

## add ratio exercised stock options / total payments

my_dataset=Add_ratio(data_dict,'ratio_exer_stock_total','exercised_stock_options','total_payments',operator='/')

## add ratio from poi to this person / total email received

my_dataset=Add_ratio(data_dict,'ratio_from_poi_email_received','from_poi_to_this_person','to_messages',operator='/')

## add ratio from this person to poi/ total email sent
my_dataset=Add_ratio(data_dict,'ratio_to_poi_email_sent','from_this_person_to_poi','from_messages',operator='/')

## add ratio exercised stock options / total stock
my_dataset=Add_ratio(data_dict,'ratio_exer_stock_total_stock','exercised_stock_options','total_stock_value',operator='/')

#print my_dataset
features_list.append('ratio_to_poi_email_sent')
features_list.append('ratio_from_poi_email_received')
features_list.append('ratio_exer_stock_total')
features_list.append('ratio_exer_stock_total_stock')

print features_list

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)



labels, features = targetFeatureSplit(data)

######### My code #################


from sklearn.preprocessing import MinMaxScaler


## scale all features
scaler = MinMaxScaler()
scaler.fit(features)
features=scaler.transform(features)

## Select best features
from sklearn.feature_selection import SelectKBest
selector= SelectKBest(chi2, k='all')
features = selector.fit_transform(features, labels)
Scores_features={}
j=0
for i in features_list[1::]:
    Scores_features[i]=selector.scores_[j]
    j+=1

print '\nScores for features:'
j=1
for k, v in sorted(Scores_features.items(), key=itemgetter(1), reverse=True):
    print '({}) {} {}'.format(j,k,v)
    j+=1


#from sklearn.feature_selection import chi2
#from sklearn.feature_selection import f_classif

#import math

#features = np.asarray(features)
#print 'Feature dimensions: {}'.format(features.shape)


#print 'New feature dimensions: {}'.format(features.shape)

#features = features.tolist()
#### TODO

##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##
j=1
print '\nfeatures list:\n'
for f in features_list :
    print '({}) {}'.format(j,f)
    j+=1
Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_1.png", f1_name=features_list[1], f2_name=features_list[2])
Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_1.png", f1_name=features_list[2], f2_name=features_list[3])
Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_1.png", f1_name=features_list[4], f2_name=features_list[6])
Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_1.png", f1_name=features_list[8], f2_name=features_list[7])
Draw(labels, features, features_list=features_list,  mark_poi=False, name="scatter_1.png", f1_name=features_list[10], f2_name=features_list[9])

#for i in my_dataset:
#    print i
#    print my_dataset[i]['ratio_from_poi_email_received']
#    print my_dataset[i]['ratio_to_poi_email_sent']

#print type(features[0])
#print type(int(features[0][0]))
#print type(labels)
#print labels

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.

##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC


#clf = make_pipeline(PCA(n_components=2), KNeighborsClassifier(n_neighbors=3))
#&clf.fit(features_test, labels_test)
#pred_test = unscaled_clf.predict(X_test)



#clf = DecisionTreeClassifier(random_state=0)
#clf = GaussianNB()
clf=KNeighborsClassifier(n_neighbors=3)
#clf = SVC()


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)



##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV



#parameters ={'n_neighbors':[4,5,6]}
#knc = KNeighborsClassifier()
#clf = GridSearchCV(knc,parameters)
#clf.fit(features_train,labels_train)
#print sorted(clf.cv_results_.keys())


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)