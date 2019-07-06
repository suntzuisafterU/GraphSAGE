from __future__ import print_function
import json
import numpy as np
import math as m

from networkx.readwrite import json_graph
from argparse import ArgumentParser

import matplotlib.pyplot as plt

def distance(x1,y1,x2,y2):
    """ Euclidean distance between points.
    """
    return m.sqrt((x1-x2)**2 + (y1-y2)**2)

if __name__ == '__main__':
    print("Loading data...")
    G = json_graph.node_link_graph(json.load(open("example_data" + "/ppi-G.json")))
    labels_x = json.load(open("example_data" + "/just_x2_class_map.json"))
    labels_x = {int(i):l for i, l in labels_x.items()}
    
    labels_y = json.load(open("example_data" + "/just_y2_class_map.json"))
    labels_y = {int(i):l for i, l in labels_y.items()}
    
    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n]['test']]
    #all_ids = [n for n in G.nodes()]
    
    train_labels_x = np.array([labels_x[i] for i in train_ids])
    test_labels_x = np.array([labels_x[i] for i in test_ids])
    #all_labels_x = np.array([labels_x[i] for i in all_ids])
    
    train_labels_y = np.array([labels_y[i] for i in train_ids])
    test_labels_y = np.array([labels_y[i] for i in test_ids])    
    #all_labels_y=np.array([labels_x[i] for i in all_ids])
    #if train_labels.ndim == 1:
    #    train_labels = np.expand_dims(train_labels, 1)
    #i=0
    #for val in train_labels:
      #  print(str(i)+" " + str(val))
       # i=i+1
    #train_labels.astype(np.int64).reshape(train_lables.size,1)
    #tr_labels = np.array([val[0] for val in train_labels],dtype=np.int64)
    print("running","unsup-example_data/graphsage_meanpool_small_0.000010")

    embeds = np.load("unsup-example_data/graphsage_meanpool_small_0.000010" + "/val.npy")
    id_map = {}
    with open("unsup-example_data/graphsage_meanpool_small_0.000010" + "/val.txt") as fp:
        for i, line in enumerate(fp):
            id_map[int(line.strip())] = i
    train_embeds = embeds[[id_map[id] for id in train_ids]] 
    test_embeds = embeds[[id_map[id] for id in test_ids]] 
    #all_embeds= embeds[[id_map[id] for id in all_ids]] 
    print("Running regression..")
    #run_regression(train_embeds, train_labels, test_embeds, test_labels)
    from sklearn.linear_model import LinearRegression 
    lin = LinearRegression()   
    from sklearn.preprocessing import PolynomialFeatures 
    from sklearn.externals import joblib
    poly = PolynomialFeatures(degree = 2)
    print("Running Regression for X-Axis")
    X_poly = poly.fit_transform(train_embeds)
    print("Fitting for X-Axis")
    lin.fit(X_poly,train_labels_x)
    joblib.dump(lin,'save_x_model.pkl')
    poly_for_x=joblib.load('save_x_model.pkl')
    PRE=poly_for_x.predict(poly.fit_transform(test_embeds))
    #all_x=poly_for_x.predict(poly.fit_transform(all_embeds))
    i=0
    for val in PRE:
        print(str(val)+" "+str(test_labels_x[i]))
        i=i+1
        if i==100:
            break
    
    print("Done for X-Axis")
    
    from sklearn.linear_model import LinearRegression 
    lin_y = LinearRegression()   
    from sklearn.preprocessing import PolynomialFeatures 
    poly_y = PolynomialFeatures(degree = 2)
    Y_poly = poly_y.fit_transform(train_embeds)
    print("Runnign Regression for Y-Axis")
    print("Fitting for X-Axis")
    lin_y.fit(Y_poly,train_labels_y)
    joblib.dump(lin_y,'save_y_model.pkl')
    poly_for_y=joblib.load('save_y_model.pkl')
    PRE_y=poly_for_y.predict(poly.fit_transform(test_embeds))
    #all_y=poly_for_y.predict(poly.fit_transform(all_embeds))
    i=0
    for val in PRE_y:
        print(str(val)+" "+str(test_labels_y[i]))
        i=i+1
        if i==100:
            break
    print("Done for Y-Axis")

    data = zip(PRE, PRE_y, test_labels_x, test_labels_y)
    distance_data = [distance(x1,y1,x2,y2) for x1,y1,x2,y2 in data]
    np.mean(distance_data) #use other np.<statistics function> calls here.

    fig=plt.figure()
    plt.scatter(test_labels_x,test_labels_y,color='blue')
    plt.scatter(PRE,PRE_y,color='red')
    plt.xlabel('X-Coordinate')
    plt.ylabel('y-Coordinate')
    plt.show()
    fig.savefig('plot.png')

