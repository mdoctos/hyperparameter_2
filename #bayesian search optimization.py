#bayesian search optimization
#using skopt.gp_minimize
import pandas as pd
import numpy as np
from sklearn import ensemble
from sklearn import metrics 
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing 
from sklearn import pipeline

from functools import partial
from skopt import space
from skopt import gp_minimize

def optimize(params ,param_name ,x,y):
    params = dict(zip(param_name ,params))
    model = ensemble.RandomForestClassifier(**params)
    kf= model_selection.StratifiedKFold(n_splits = 5)
    accuracies = []
    for idx in kf.split(X =x ,y=y): #idx is index
        train_idx, test_idx =idx[0] , idx[1]
        xtrain = x[train_idx] #spliting the data
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]


        model.fit(xtrain ,ytrain)
        pred = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest,pred) #calcuting accurracy
        accuracies.append(fold_acc)

    return -1.0 *np.mean(accuracies)




if __name__ =="__main__":



    
    df = pd.read_csv("E:\Microsoft VS Code/train.csv")

    X = df.drop("price_range" , axis = 1).values
    y = df.price_range.values


    param_space = [
        space.Integer(3,15, name = "max_depth"),
        space.Integer(100 ,600 , name = "n_estimators"),
        space.Categorical(["gini","entropy"], name = "criterion"),
        space.Real(0.01 , 1 ,prior ="uniform" , name = "max_features"),
    ]
    param_name = [
        "max_depth", 
        "n_estimators",
        "criterion",
        "max_features"

    ]

    optimization_fucntion =partial(
                        optimize ,                                     
                        param_name =param_name,
                        x= X,
                        y=y, 
                                   
                                   )
    result =gp_minimize(optimization_fucntion , dimensions = param_space,
                        n_calls = 15 ,
                        n_random_starts = 10 ,
                        verbose=10 ,
                        )
    print(
        dict(zip(param_name, result.x))
    )
