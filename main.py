# -*- coding: utf-8 -*-
# Import pandas to read xlsx
import pandas as pd
import graphviz 
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import tree

# Xlsx file name
file = "decision_tree.xlsx"

# Load spreadsheet
xl = pd.ExcelFile(file)

et = ExtraTreesClassifier(n_estimators=20, max_depth=None, min_samples_split=2, random_state=0)

# Print the sheet names

# Load a sheet into a DataFrame by name: df
df = xl.parse('table1')

columns = ['sex','age','education','playinstrument','Extrversion', 'greebleness','Conscientiousness', 'Neuroticism', 'Openness', 'legislative', 'executive', 'judicial', 'global', 'local', 'liberal', 'conservative', 'hierarchical', 'monarchic', 'oligarchic', 'anarchic', 'internal', 'external']

i = 1
while(i<27):
    if i < 23:
        index = 'G'+str(i)
    else:
        index = 'M'+str(i-22)
    labels = df[index].values
    features = df[list(columns)].values

    
    
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf = clf.fit(features, labels)
    dot_data = tree.export_graphviz(clf, out_file=None) 
    graph = graphviz.Source(dot_data)
    graph.render(str(index)) 
    
    et_score = cross_val_score(clf, features, labels, n_jobs=-1, scoring = 'f1').mean()
    
    #print labels
    #print features
    print(index + ':' + str(et_score))
    i+=1
