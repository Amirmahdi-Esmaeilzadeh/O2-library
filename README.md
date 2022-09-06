# O2-library

📚a library for fitting KNN,ANN,DBSCAN,LOGISTICREGRESSION,SVM on images(with reading the images and preprocessing func))


🐱‍💻Functions:
🤖(images,image_labels)=read_dataset('address of dataset',[labels in numeric],need=if the main label is string True else False,ilabel=[if need: the real string labels(the folder names)],asize=(size of each image you want))

🤖(model,x_test,y_test)=fit_model(images,image_labels,algorithm='between knn, logistic regression,svm',k=if algorithm is knn)

🤖recommended_k_for_knn=k_recommendation(images,image_labels)

🤖prediction_of_model=dbscan(images,epsilon,min_samples)

🤖(model,history of model)=ann(x_train,y_train,x_test,y_test,input_dims=(),nourons=[],activations=[],optimizer='adams',loss='sparse_categorical_crossentropy',epochs=30,batch_size=32,resize=False,size=())

🤖cheatsheet(function you want help about it)


👨🏻‍💻Example:
fitting svm on mnist:
import O2
import cv2
(images,imlabels)=O2.read_dataset('D:/robotic/Python/trainingSet/',list(range(0,10)),0,asize=(28,28))
(model,x_test,y_test)=O2.fitmodel(images,imlabels,'svm')
model.predict(x_test,y_test)

👨🏽‍🏫how to download and use:

download the O2.py and place it in the kernel folder in the code import it like this
import O2
and enjoy it...

by amirmahdi esmaeilzadeh
