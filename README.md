# PVSM
Plant Village Small Model

This is the smaller dataset where I have used 10 categories to identfy the plant disease with 96% accuracy.
SInce I use RTX 4070 I cant increase buffer size and batch size from 200/8. If you have better GPU you can do so .
Techniques like EarlyStopping,ReduceLROnPlateau, LearningRateScheduler and Data augumentation is used and model is saved with .h5 format
Fast API file contains the loaded model which will serves a deep learning model for classifying images into one of several classes related to pepper, potato, and tomato plant diseases.
The model takes an image as input, processes it, and returns the predicted disease class along with the confidence of the prediction.



