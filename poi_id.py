#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data



##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##



## function : data cleaning
## content: remove item from dictionary
## return: data dictionary with removed items


def data_cleaning(data_dict,list_items_to_remove):
    for i in list_items_to_remove:
        print '---> Delete: {}'.format(i)
        del data_dict[i]

## function: draw
## content: draw a scatter plot with 2 features for axis and a different plot color for poi or not poi
## return: nothing

from matplotlib import pyplot as plt
import numpy as np

def Draw(pred, features, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c"]
    for i,j in enumerate(pred):
        ax.scatter(int(features[i][0]),int(features[i][1]), color = colors[int(pred[i])])     

    point_1=plt.scatter(0,0, color = colors[0]) 
    point_2=plt.scatter(0,0, color = colors[1]) 
    
    ax.legend([point_1,point_2], ["not poi", "poi"])
    
    plt.legend(loc=4)
    plt.grid(True)
    
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()

## function: feature_NaN
## content: count the number of NaN value for each value a data dictionnary
## return: a dictionary with the feature as a key and the number of NaN as value

def Draw_bar(dict_input,name='bar_1.png',title='bar_chart'): 
    
    poi=[]
    not_poi=[]
    x_label=dict_input.keys()
    
    for i in dict_input:
        poi.append(dict_input[i]['poi'])
        not_poi.append(dict_input[i]['not_poi'])
    
    N = len(dict_input)
    
    ind = np.arange(4*N,step=4)  # the x locations for the groups
#    print 'ind:{}♥'.format(ind)
    width = 1.5       # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, poi, width, color='r')
       
      
    rects2 = ax.bar(ind + width, not_poi, width, color='y')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x_label,rotation=90)#('G1', 'G2', 'G3', 'G4', 'G5'))
    

    ax.legend((rects1[0], rects2[0]), ('poi', 'not_poi'))

    
    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 0.95*height,'%d' % int(height),ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.savefig(name)
    plt.show()

def feature_NaN(data_dict):
    dict_NaN={}

    for i in data_dict.keys():
        for j in data_dict[i]:
            if data_dict[i][j]=='NaN':
                if j in dict_NaN.keys():
                    dict_NaN[j]+=1
                else:
                    dict_NaN[j]=1
    return dict_NaN

## function: POI_NaN
## content: count the number of NaN value for each value a data dictionnary
## return: a dictionary with the feature as a key and the number of NaN as value

def feature_NaN_poi(data_dict, NaN=True, poi=True):
    dict_NaN_poi={}

    for i in data_dict.keys():
        for j in data_dict[i]:
            if (data_dict[i][j]=='NaN' and NaN) or (data_dict[i][j]!='NaN' and not NaN):
                if j in dict_NaN_poi.keys():
                    if data_dict[i]['poi']==True:
                                dict_NaN_poi[j]['poi']+=1
                    else:
                        dict_NaN_poi[j]['not_poi']+=1
                else:
                    dict_NaN_poi[j]={}
                    if data_dict[i]['poi']==True:
                        dict_NaN_poi[j]['poi']=1
                        dict_NaN_poi[j]['not_poi']=0
                        
                    else:
                        dict_NaN_poi[j]['poi']=0
                        dict_NaN_poi[j]['not_poi']=1
    return dict_NaN_poi



## function: feature_analysis
## content: analyse one feature of one dictionary in order to find min, max or NaN Value
## return: a list of dictionary with min/max or NaN names + value

def feature_analysis(data_dict, feature,analysis='max'):
    dict_feature={}
    for i in data_dict.keys():
        if data_dict[i][feature] not in ['NaN','None','TOTAL'] :
            dict_feature[i]=data_dict[i][feature]
    
    v=list(dict_feature.values())
    k=list(dict_feature.keys())
 
    return k[v.index(max(v))],max(v)
    
## function: Remove_NaN_Person 
## content: Remove person with a specified feature == NaN
## return: nothing

def Remove_NaN_Person(data_dict, feature, number_nan=6):
    list_delete=[]
    
    for i in data_dict.keys():
        to_delete=0
        #print i
        for f in feature:
            if data_dict[i][f] == 'NaN':
                #print f,'ok'
                to_delete+=1
             #else:
                #print f,'nok'
        if to_delete>=number_nan:
            #print 'delete',i
            list_delete.append(i)
            
    data_cleaning(data_dict,list_delete)

def Count_NaN(data_dict, feature):
    list_person={}
       
    for i in data_dict.keys():
        nan_count=0
        for f in feature:
             if data_dict[i][f] == 'NaN':
                 nan_count+=1
        list_person[i]=nan_count
    return list_person

def List_NaN(data_dict, feature):
    list_feature=[]
    for f in feature:
        if data_dict[f] == 'NaN':
            list_feature.append(f)
    return list_feature
        

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### baseline features_list = ['poi','salary'] # You will need to use more features
#features_list = ['poi','salary', 'total_stock_value']
features_list = ['poi','exercised_stock_options','total_stock_value', 'total_payments','from_poi_to_this_person','shared_receipt_with_poi','from_this_person_to_poi']

                             
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

## I remove TOTAL from list of person
data_cleaning(data_dict,['TOTAL'])

## I remove person with NaN in selected features
Remove_NaN_Person(data_dict,features_list[1::],number_nan=len(features_list[1::]))

print '\n## Number of persons after cleaning: {}\n'.format(len(data_dict))


for i in data_dict.values()[0]:
    print 'Maximum value for {}: {}'.format(i,feature_analysis(data_dict,i,analysis='max'))


print '\nFeatures for max Salary:{}'.format(data_dict[feature_analysis(data_dict,'salary',analysis='max')[0]])


##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
dict_nan=Count_NaN(data_dict, features_list[1::])

print '\nNumber of NaN per person:'
j=1
for k, v in sorted(dict_nan.items(), key=itemgetter(1), reverse=True):
    print '({}) {}:\tNAN={}/{}:{}'.format(j,k,v,len(features_list[1::]),List_NaN(data_dict[k], features_list))
    j+=1

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)



labels, features = targetFeatureSplit(data)

##%%%%%%%%%%%%%%% MY CODE %%%%%%%%%%%%%%##
Draw(labels, features,  mark_poi=False, name="scatter_1.png", f1_name=features_list[1], f2_name=features_list[2])
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
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)