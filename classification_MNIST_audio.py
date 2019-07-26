'''
MNIST for audio recordings. We classify the MFCC features of 2000 audio recordings, by 
4 subjects. Each subject pronounces each of the digits {0,1, ...9} , 50 times, hence
4*10*50 = 2000 audio signals. We extract the MFCCfeatures from them, and use 
these features to build a classification model for 10 class classification of
audios pertaining to pronouncing digtis.

'''

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import scipy
#import pyaudio
#import struct
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import preprocessing
from sklearn import model_selection # for command model_selection.cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
#from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
#from scipy.stats.stats import pearsonr

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from scipy import signal
from scipy.io import wavfile


import os

'''
#Below, path is where we have the the folder 'recordings' with all the audio samples
#You need to modify the path acording to where you saved the folder 'recordings'
#note that below, k will change acordingly as your path name and hence fullfilenames
#in our case k = 98
'''
path = r'C:\Users\PC\Documents\Important documents\MNIST audio' #see the note above!!!

'''The following function resamples every row of a matrix and produces, for each original
row, a resampled row of num dimensions. The reason to do so is that for different audio 
signals, the numbers of time indexed samples are different'''
def resample_matrix(F,num):
    G=np.empty((F.shape[0], num))
    for i in range(F.shape[0]):
     G[i]=scipy.signal.resample(F[i], num, t=None, axis=0, window=None)
    return G
    

'''Below, we extract features using Py Audio Analysis, resample all feature vectors, and 
store them in the list feat_list'''
files=[] 
feat_list=[] #list of (resampled) feature matrices for each signal
Fs_lst=[]
x_lst=[]
for r, d, f in os.walk(path): #root, dir, file
    for file in f:
        if '.wav' in file:
            full_filename=os.path.join(r, file)
            files.append(full_filename)
            [Fs, x] = audioBasicIO.readAudioFile(full_filename)
            Fs_lst.append(Fs)
            x_lst.append(x)
            F, f_names = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs);
            G=resample_matrix(F,num=15) #see the definitionog resample_matrix above
            feat_list.append(G)
            
'''Below, we extract the MFCC's from the extracted and then resampled features above 
.For each i, MFCC_list[i] gives for the i-th audio, all 13 feature vectors, each of them 15-dimensional 
resampled values of the original feature vectors'''  
MFCC_list=[]
for i in range(len(feat_list)):  
    MFCC= feat_list[i][8:21,:] #9th to 21-st features are the MFCC coeffs
    MFCC_flat=np.ndarray.flatten(MFCC)#flatening the array, but are we destroying time series structure?
    MFCC_list.append(MFCC_flat)
MFCC_array=np.asarray(MFCC_list)
    #print(MFCC_array)
    
    
'''Below, we prepare the y values for the classifier; files[i][k] gives us the number
pronounced for the i-th audio file, where 0 <= i <= 1999. Val. of k depends on the 
indivisual file path and fullfilename depending on your machine.
So, y stores all the numbers pronounced''' 
y=[] 
k = 98 # k is the index of the string files[i] so that files[i][k] gives us the no. pron.
for i in range(len(files)):
    y.append(float(files[i][k]))
#y=np.asarray([y]).T #way to convert y into an array of d by 1 dim , and not just d -dim
y=np.asarray(y)    
#print(y)    

speakers=[]
for i in range(len(files)):
    if 0 <= i%200 <= 49:
        speakers.append('Jackson')
    elif 50 <= i%200 <= 99:
        speakers.append('Nicolas')
    elif 100 <= i%200 <= 149:
        speakers.append('Theo')
    else:
        speakers.append('Yweweler')        
speakers=np.asarray(speakers)  #converting to np array as it'll be needed to 
#compare the relative values in pd dataframe  
    
'''Writing the np arrays into a pd DataFrame with specified columns names, first we convert
each of the three np arrays, MFCC, speakers and numbers_pronounced into seperate data frame
and then horizontally concat them. If we'd not have converted the previous lists into np 
then later, seaborn would throw a TypeError: unsupported operand types for +: 'int' and 'str' '''  

columns_dat=['f'+str(i) for i in range(0,MFCC_array.shape[1])]
dat_MFCC=pd.DataFrame(MFCC_array, columns=columns_dat)
dat_speakers=pd.DataFrame(speakers, columns=['speakers'])
numbers_pronounced=pd.DataFrame(y, columns=['numbers_pronounced'])
df=pd.concat((dat_MFCC, dat_speakers, numbers_pronounced), axis=1)


'''Below, we do some analysis of the data'''
Cov=np.cov(MFCC_array.T)
Corr_coef=np.corrcoef(MFCC_array.T)
#Below we see find that the many corr coef value will lie between their mean - sd and mean + sd
print("Most correlation coeffs. among features are between \n" + str(np.mean(abs(Corr_coef))- np.std(abs(Corr_coef))) \
      + ' and ' +  str(np.mean(abs(Corr_coef)) + np.std(abs(Corr_coef))))

'''Because of not so high correlation among features, and high variances of many featutes,
we don't do PCA or dimensionality reduction for now'''


'''Below, we do some data visualization to see if there're at least some visual features
significantly different for the numbers 0,1,...9. Turns out several features are indeed
different, e.g. f_30, f_100,f_150, among many others.'''
sns.barplot(x='numbers_pronounced',y='f30',data=df);plt.show()
sns.barplot(x='numbers_pronounced',y='f100',data=df);plt.show()
sns.barplot(x='numbers_pronounced',y='f150',data=df);plt.show()
#sns.barplot(x='numbers_pronounced',y='f30',hue='speakers',data=df);plt.show()


'''train test split, and rescaling of the original data, but no PCA here (done later)'''
X = MFCC_array
Y = y
test_size = 0.20
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, shuffle=True)
X_train=preprocessing.scale(X_train)
X_test=preprocessing.scale(X_test)


'''model building'''
models = []
models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

'''Below, we test the above models with default values to get idea of accuracies; for our
data, we observe that SVM has lowewt bias and also lowest variance; hence in the
next section, we optimize the hyperparemeters of the SVM and not ohter models'''
results = []
names = []
scoring= 'accuracy'      
for name, model in models:
    #below, we do k-fold cross-validation
	 kfold = model_selection.KFold(n_splits=10, shuffle=True) 
	 cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	 results.append(cv_results)
	 names.append(name)
	 msg = "%s: %f (%f)" % ('mean and std of the cv accuracies for ' + name,  cv_results.mean(),  cv_results.std() );print(msg)
    
    
'''Below, we perform SVM with the default parameters'''   
SVM=SVC()
SVM.fit(X_train, Y_train)
pred_train= SVM.predict(X_train);
print('training accuracy for SVM, before PCA and hyp. opt. is '  +  str(accuracy_score(Y_train, pred_train)) )
predictions = SVM.predict(X_test)
print('test accuracy for SVM is before PCA and hyp. opt. is '  +  str (accuracy_score(Y_test, predictions))  )
print('confusion matrix for SVM before PCA and hyp. opt. is \n' + str(confusion_matrix(Y_test, predictions)) )
print('detailed classification results for test data, before PCA and hyp. opt. are \n' + str(classification_report(Y_test, predictions)) )    

'''Below, we perform SVM with grid search'''
print('START GRID SEARCH TO OPTIMIZE HYPERPARAMETERS OF SVM, WHICH ALREADY PERFORMED WELL WITH DEFAULT PARAMETERS \n')
parameters = {'kernel':('linear', 'rbf'), 'C':(10,1,0.25,0.5),'gamma': (0.001,'auto', 0.01, 1)}
clf_GS = GridSearchCV(estimator=SVM, param_grid=parameters, scoring=scoring, cv=5)
clf_GS.fit(X_train,Y_train)
best_accuracy=clf_GS.best_score_
best_params=clf_GS.best_params_
print("The best parameters for accuracy before PCA are: \n" + str (best_params) )



'''Below, we make prediction with the hyperparameter optimized SVM'''
print('')
predictions_opt = clf_GS.predict(X_test)
print( 'test accuracy for hyper parameter optimized SVM, before PCA is \n'  +  str (accuracy_score(Y_test, predictions_opt))  )
print('confusion matrix for SVM on test data, before PCA is \n' + str(confusion_matrix(Y_test, predictions_opt)) )
print('detailed classification results for test data, before PCA are \n' + str(classification_report(Y_test, predictions_opt)) )    

'''We've already gotten good results, but now we redo everything with PCA, just in case we
obtained good results because of high feature dimension and overfit happened. But I think
chance of that heppening is slim as the no. of samples are large enough compared to the
features, and features don't have high correlation'''
print('PERFORMING PCA \n')
pca = PCA(n_components=25)
#pca = PCA(.95)
pca.fit(X_train)
X_train_PCA = pca.transform(X_train)
X_test_PCA = pca.transform(X_test)


print('START GRID SEARCH TO OPTIMIZE HYPERPARAMETERS OF SVM, FOR THE FEATURES AFTER THE PCA \n')
SVM=SVC()
SVM.fit(X_train_PCA, Y_train)
pred_train_PCA= SVM.predict(X_train_PCA);
print('training accuracy for SVM, after PCA is '  +  str(accuracy_score(Y_train, pred_train_PCA)) )

'''Below, we perform SVM with grid search on the data, after the PCA'''
parameters = {'kernel':('linear', 'rbf'), 'C':(10,1,0.25,0.5),'gamma': (0.001,'auto', 0.01, 1)}
clf_GS_PCA = GridSearchCV(estimator=SVM, param_grid=parameters, scoring=scoring, cv=5)
clf_GS_PCA.fit(X_train_PCA,Y_train)
best_accuracy_PCA=clf_GS_PCA.best_score_
best_params_PCA=clf_GS_PCA.best_params_
print('The best parameters for accuracy after PCA are: \n' + str (best_params_PCA) )

'''Below, we make prediction with the hyperparameter optimized SVM, after PCA'''
predictions_opt_PCA = clf_GS_PCA.predict(X_test_PCA)
print('test accuracy for hyper parameter optimized SVM is \n'  +  str (accuracy_score(Y_test, predictions_opt_PCA))  )
print('confusion matrix for SVM, after PCA, is \n' + str(confusion_matrix(Y_test, predictions_opt_PCA)) )
print('detailed classification results, after PCA, are \n' + str(classification_report(Y_test, predictions_opt_PCA)) )    



'''Below, we visualize the spectogram'''
#sample_rate, samples = wavfile.read('doremi.wav')
k=200
sample_rate, samples = wavfile.read(files[k])
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

'''Below, we plot the spectograms for 6's and 8's, as they're often confused'''
l=1
for k in range(l*50+1, l*50+201):
    sample_rate, samples = wavfile.read(files[k])
    frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

l=2
for k in range(l*50+1, l*50+201):   
    f, t, Sxx = signal.spectrogram(x_lst[k], Fs_lst[k])
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show() 
    
   
