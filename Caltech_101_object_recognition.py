# Caltech 101 Dataset - Object recognition using Bag of features 
# Kishan S Athrey

from PIL import Image
import numpy as np
from scipy import misc
import csv
import pandas as pd
import cv2
from sklearn.cluster import KMeans
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import time
from sklearn.naive_bayes import MultinomialNB
import os


print 'Run the code as it is. It performs the generation of vocabulary, training using RF classfier and testing'
print 'Comment out the randomforest model and uncomment naive bayes model at line 176 and 177 to run NB classifier'
print 'Currently the number of cluster centers are 100. This code takes around 14 min to give the result'
print 'For better performance run the code with \'number of centroids = 200. It takes about than 45 min to give the result' 


############################################### Functions necessary for BoF model##############################################################################
# Function to read training images from the train folder
def read_training_images(i):  
    image_list = []
    for filename in glob.glob(os.getcwd()+'\\Reduced_data_set_caltech_101\\train\\'+str(i)+'/*.png'):
        #im=Image.open(filename)
        im=misc.imread(filename,flatten=1)
        image_list.append(im)
    return image_list
    
# Function to read testing images from the test folder
def read_testing_images(i):  
    image_list = []
    for filename in glob.glob(os.getcwd()+'\\Reduced_data_set_caltech_101\\test\\'+str(i)+'/*.png'):
        #im=Image.open(filename)
        im=misc.imread(filename,flatten=1)
        image_list.append(im)
    return image_list

# Preprocess the images. Converting color image to grayscale. Resize it to 100*100 pixels. Stack all images of a given folder/class into a 3D matrix
def preprocess_images(image_list):
    gray_image = []
    for i in range(len(image_list)):
        gray_image.append(misc.imresize(np.array(image_list[i]),(100,100)))    
    size_of_each_frame = gray_image[1].shape
    number_of_frames = len(gray_image)
    Frames =  np.zeros((size_of_each_frame[0],size_of_each_frame[1],number_of_frames))
    for i in range(len(gray_image)):
        Frames[:,:,i] = gray_image[i]
    return Frames

# Writes the size of the descriptors of each image to a csv file
def write_size_of_descriptors_to_csv(size_of_descriptors,mode):
    with open('Feature_descriptor_sizes_'+str(mode)+'.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',lineterminator='\n')
        spamwriter.writerow([size_of_descriptors])
    return
# Writes the computed feature descriptors to a csv file
def Write_features_to_csv(Feature_list,mode):
    with open('SURF_Features_'+str(mode)+'.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',lineterminator='\n')
        spamwriter.writerows(Feature_list)

# Computes the SURF descriptors on each image by taking the 3D stack of images as input. 
# Each of the feature descriptors is stacked vertically into a matrix and saved into a csv file
def get_SURF_descriptors(Frame_set,mode):  
    des = 0  
    kp = 0
    feature_descript = []
    dim_of_frames = Frame_set.shape
    no_of_frames = dim_of_frames[2]        
    for i in range(0,no_of_frames):         
        each_frame = np.uint8(Frame_set[:,:,i])
        surf = cv2.SURF(25) # Creates the SURF object with Hessian threshold as 25
        kp,des = surf.detectAndCompute(each_frame,None) # Computes the interest points and descriptors
        write_size_of_descriptors_to_csv(des.shape[0],mode) # Writes the size of the descriptors into the csv file
        feature_descript.extend(des)
        matrix_stack = feature_descript[0] 
    for j in range(len(feature_descript)-1):        
        matrix_stack = np.vstack((matrix_stack,feature_descript[j+1])) # Stacking the features vertically into a matrix   
    Write_features_to_csv(matrix_stack,mode) # Writes the stacked feature descriptors into a file
    return matrix_stack 

# Generates the vocabulary or the code book or code visual word dictionary
def generate_vocabulary(no_of_clusters,number_of_iterations,mode): 
    print 'Building_Vocabulary' 
    length_of_image_list = []         
    for m in range(number_of_iterations):    
        img_list = read_training_images(m) # Reading images
        length_of_image_list.append(len(img_list))
        preprocessed_image_set = preprocess_images(img_list) # Preprocessing images
        descriptors = get_SURF_descriptors(preprocessed_image_set,mode) # Getting descriptors  
    # Reading the Feature descriptor file using pandas dataframe     
    Feature_descriptors = pd.read_csv('SURF_Features_'+str(mode)+'.csv',header = None)
    print 'Performing clustering'
    # Using sklearn Kmeans clustering, cluster the feature descriptors to generate code book. Each cluster center is a code word
    kmeans = KMeans(n_clusters=no_of_clusters,init='k-means++').fit(Feature_descriptors)
    return kmeans,length_of_image_list

# This function selects the corresponding set of feature descriptors of a particular image, so that it can be used in training  
def read_features_from_each_image(descriptors_for_train,start_value,end_value):
    temp = descriptors_for_train[start_value:end_value,:]
    return temp
    
# This function takes each set of feature descriptors per image and assigns those descriptors to the nearest centroids of the K-means clusters    
def compute_nearest_centroids(descripts_per_image,centroids):
    nearest_centroids = centroids.predict(descripts_per_image)
    return nearest_centroids
    
# This function does vector quantization and compute a normalized histogram vector per image 
def compute_feature_histogram_of_each_image(indexes,no_of_centroids):
    histogram = []
    no_of_points = indexes.shape[0] 
    for k in range(no_of_centroids):
        histogram.append(np.count_nonzero(indexes==k)/float(no_of_points))
    return histogram
    
# Writes the histograms into a csv file    
def write_histograms_to_csv(histograms,mode):
    with open('Histograms_'+str(mode)+'.csv', 'a') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',lineterminator='\n')
        spamwriter.writerows([histograms])
    return
# Since the dataset is unlabelled, this function creates the label for each image
def create_labels(num_of_img_per_folder):
    y_train = []
    length = len(num_of_img_per_folder)
    for n in range(length):
        temp = num_of_img_per_folder[n]
        for k in range(temp):
            y_train.append(n)
    train_label = np.array(y_train)
    return train_label

if __name__ == "__main__":
############################################## Training phase starts here ##########################################################################################
    start_time = time.time()
    print 'Preparing for training'
    mode = 'train' # Mode decides whether those functions described above access training data or testing data
    no_of_centroids = 100 # Size of the code book/ Vocabulary size - same as the value for 'K' in the K means
    no_of_iterations = 15 # Refers to number of categories

    Vocabulary,number_of_images_per_folder = generate_vocabulary(no_of_centroids,no_of_iterations,mode) # The Vocabulary model is imported
    total_number_of_imgs = sum(number_of_images_per_folder)
    print 'Vocabulary is built'

    # Feature set is being read into a pandas dataframe
    Features_set = pd.read_csv('SURF_Features_train.csv', header = None).as_matrix()
    Offset_set = pd.read_csv('Feature_descriptor_sizes_train.csv',header = None).as_matrix()

    # Preparing the data to perform training
    print 'Preprocessing before training'        
    start = 0
    end = Offset_set[0,0]

    descriptors_of_each_img = read_features_from_each_image(Features_set,start,end)
    indexes = compute_nearest_centroids(descriptors_of_each_img,Vocabulary)
    hist = compute_feature_histogram_of_each_image(indexes,no_of_centroids)
    write_histograms_to_csv(hist,mode)

    for i in range(1,Offset_set.shape[0]):
        start = end
        end = start + Offset_set[i,0]
        descriptors_of_each_img = read_features_from_each_image(Features_set,start,end)
        indexes = compute_nearest_centroids(descriptors_of_each_img,Vocabulary)
        hist = compute_feature_histogram_of_each_image(indexes,no_of_centroids)
        write_histograms_to_csv(hist,mode)
    # Reading the Histogram file (training) using pandas
    X = pd.read_csv('Histograms_train.csv',header = None)  
    Y_train = create_labels(number_of_images_per_folder)# generating training labels
    print 'Training started'
    # Random forest is used for the classification
    model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy',random_state = 42)
    # Uncomment the below line and comment out the above line to run Naive bayes classifier 

    #model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

    # Performs training
    model.fit(X,Y_train)
    print 'Training Ended'

    ######################################### Testing phase starts here ###############################################################################################
    print 'Preparing for testing'
    mode = 'test' # Mode is set to 'test'

    length_of_image_list = []
    #Test images are read from the test folder and feature descriptors are computed for testing images. These features are saved into a csv file 
    for m in range(no_of_iterations):    
        img_list = read_testing_images(m)
        length_of_image_list.append(len(img_list))
        preprocessed_image_set = preprocess_images(img_list)
        descriptors = get_SURF_descriptors(preprocessed_image_set,mode)           
        Feature_descriptors = pd.read_csv('SURF_Features_'+str(mode)+'.csv',header = None)

    # Feature descriptors of test images are read from the file into pandas dataframe
    test_Features_set = pd.read_csv('SURF_Features_test.csv', header = None).as_matrix()
    test_Offset_set = pd.read_csv('Feature_descriptor_sizes_test.csv',header = None).as_matrix()

    # Preparing the data to perform testing
    print 'Preprocessing before testing' 
    start_point = 0
    end_point = test_Offset_set[0,0]
    descriptors_of_each_image = read_features_from_each_image(test_Features_set,start_point,end_point)
    indices = compute_nearest_centroids(descriptors_of_each_image,Vocabulary)
    hist_test = compute_feature_histogram_of_each_image(indices,no_of_centroids)
    write_histograms_to_csv(hist_test,mode)
    for i in range(1,test_Offset_set.shape[0]):
        start_point = end_point
        end_point = start_point + test_Offset_set[i,0]
        descriptors_of_each_image = read_features_from_each_image(test_Features_set,start_point,end_point)
        indices = compute_nearest_centroids(descriptors_of_each_image,Vocabulary)
        hist_test = compute_feature_histogram_of_each_image(indices,no_of_centroids)
        write_histograms_to_csv(hist_test,mode)
        
    # Reading the Histogram file (test) using pandas
    X_test = pd.read_csv('Histograms_test.csv',header = None)  
    Y_test = create_labels(length_of_image_list)  # Creating the test set labels
    print 'Testing started'
    # Predicting the label using the model created during training
    Y_predicted = model.predict(X_test)
    print 'Computing performance metrics'

    # Calculates Accuracy, Specificity and Sensitivity
    Accuracy = metrics.accuracy_score(Y_test,Y_predicted)
    Confusion_mat = confusion_matrix(Y_test,Y_predicted)
    print Accuracy
    print Confusion_mat.shape
    sensitivity = []
    specificity = []
    for i in range(Confusion_mat.shape[0]):
    	TP = float(Confusion_mat[i,i])  
    	FP = float(Confusion_mat[:,i].sum()) - TP  
    	FN = float(Confusion_mat[i,:].sum()) - TP  
    	TN = float(Confusion_mat.sum().sum()) - TP - FP - FN
    	sensitivity.append(TP / (TP + FN))
    	specificity.append(TN / (TN + FP))
    	print 'Sensitivity is:'+str(sensitivity) + 'Specificity is:'+str(specificity)
    Stop_time = time.time()
    Execution_time = (Stop_time-start_time)/60.0 
    print Execution_time





















