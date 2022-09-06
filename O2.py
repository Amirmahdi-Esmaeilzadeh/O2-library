import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
def ann(x_train,y_train,x_test,y_test,settings='manual',input_dims=1,nourons=[],activations=[],optimizer='adams',loss='sparse_categorical_crossentropy',epochs=30,batch_size=32,resize=False,size=()):
    from tensorflow.keras import models,layers
    from tensorflow.keras.layers import Dense
    x_train,x_test=x_train/255,x_test/255
    if resize:
        x_train = np.reshape(x_train, size)
        x_test = np.reshape(x_test, size)
    model=models.Sequential()
    for n in nourons:
        if nourons.index(n) == 0:
            model.add(Dense(units=n,activation=activations[nourons.index(n)],input_dim=input_dims))
        else:
            model.add(Dense(n,activation=activations[nourons.index(n)]))
    model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])
    h=model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))
    return (model,h)


def read_dataset(add='',labels=[],need=0,ilabel=[],asize=(250,250)):
    #labels=[1,0]
    images=[]
    imlabels=[]
    c=1

    #add="D:/robotic/Python/hotdog-nothotdog/hotdog-nothotdog/train/"
    for a in labels:
        if need:
            z=ilabel[labels.index(a)]
        else:
            z=str(a)
        print("\nreading",z)
        for image in glob(add+z+"//*"):
            print('.',end='')
            img=cv2.imread(image)
            img=cv2.resize(img, asize)
                
            img=img.flatten()
            imlabels.append(a)
            images.append(img)

            if(c==4000):
                print('\nalmost done\n')
            c+=1

    
    images=np.array(images)
    imlabels=np.array(imlabels)

    print('\nDetails:')
    print('shape:',images.shape)
    print('pixel colors:',images.shape[1]/(asize[0]*asize[1]))
    if images.shape[1]/(asize[0]*asize[1]):
        print('Colored pictures')
    else:
        print('gray scale or binary images')
    print(images.shape[0],'pictures')
    print('size of images:',asize)
    print('Flattened: YES')
    return (images,imlabels)
def fitmodel(images,imlabels,algorithm,k=0):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(images,imlabels,test_size=0.2,shuffle=1)
    if(algorithm=='svm' or algorithm=='SVM'):
        from sklearn.svm import SVC
        model=SVC()
        model.fit(x_train,y_train)     
    elif(algorithm=='logistic regression' or algorithm=='Linear Regression'):  
        from sklearn.linear_model import LogisticRegression
        model=LogisticRegression()  
        model.fit(x_train, y_train)
    elif(algorithm=='knn'):

        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(x_train,y_train)


    else:
        print("not available yet")
    return (model,x_test,y_test)

def cheatsheet(func):
    if func=="read_dataset":
        print("read_dataset(add='',labels=[],need=0,ilabel=[],asize=(250,250))")
        print("Used to read picture dataset and preprocess")
        print("add: adress of the dataset")
        print("labels: list of labels(if it is in string format turn it into num e.g: ['hello','goodbye'] > [1,0])")
        print('need: if the folder adress is not equal to labels')
        print('ilabel: the adress of the file(if the folder adress is not equal to labels)')
        print('asize: the size of the images you want to have')
        print('it will return a tuple containing the preprocessed images and labels')
    elif func=='fitmodel':
        print('images: the images you want to fit with the model')
        print('imlabel: the labels you want to fit with the model')
        print('algorithm: the algorithm you want to use e.g: knn or svm')
        print('k for knn')
        print('it will return the model(that you can predict with model.predict() and have the score with model.score())')
    elif func=='k_recommendation':
        print('images:the images you want to fit with the model')
        print('imlabel: the labels you want to fit with the model')
        print('returns best k for knn model')
    else:
        print('images:the images you want to fit and predict')
        print('eps and m for epsilon and min sample of dbscan')

def k_recommendation(images,imlabels):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(images,imlabels,test_size=0.2,shuffle=1)    
    from sklearn.neighbors import KNeighborsClassifier
    error_rate = []
    for i in range(0,50):

        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train,y_train)
        pred = knn.predict(x_test)
        error_rate.append(np.mean(pred != y_test))
    return error_rate.index(min(error_rate))

def dbscan(images,eps=29,m=3):
    from sklearn.cluster import DBSCAN
    db=DBSCAN(eps=eps,min_samples= m)
    return db.fit_predict(images)

def save(model,name='model'):
    from pickle import dump
    dump(model,'O2'+name)
    









