

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

def Draw(pred, features, features_list, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    index_1=features_list.index(f1_name)-1
    index_2=features_list.index(f2_name)-1
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c"]
    for i,j in enumerate(pred):
        ax.scatter(float(features[i][index_1]),float(features[i][index_2]), color = colors[int(pred[i])])     

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
 
    if analysis=='max':
        return k[v.index(max(v))],max(v)
    elif analysis=='min':
        return k[v.index(min(v))],min(v)
    
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
        
def Add_ratio(data_dict,new_feature_name,feature_1,feature_2,operator='/'):
    
    new_data_dict=data_dict
    
    for i in data_dict:
        if (data_dict[i][feature_1]!='NaN') and (data_dict[i][feature_2]!='NaN'):
            if operator=='/':
                data_dict[i][new_feature_name]=round((1.0*data_dict[i][feature_1])/data_dict[i][feature_2],4)
                #print data_dict[i][new_feature_name]
            
        else:
            data_dict[i][new_feature_name]='NaN'
    return new_data_dict
        
