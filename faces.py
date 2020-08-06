import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

#used to scale eigenvectors from 0 - 255
def scale(image):
    #to be used for displaying eigenfaces where pixels may be negative or greater than 255
    imin = np.min(image)
    image -= imin
    imax = np.max(image)
    image = image * (255/imax) 
    return image

def principal_c():
    #to find the variance contained in the k principal components
    total = np.sum(evalues)
    ratio = 0
    ratio = np.sum(evalues[0:k])/total
    return ratio

def eigenfaces():
    #multiply k eigenfaces by identity matrix of dimension k to produce matrix 
    #of desired eigenfaces. The transpose is taken so that B[n] will be an eigenface
    eigenfaces = (B[:, 0:k] @ np.identity(k)).T
    #reshape to image dimensions
    eigenfaces = eigenfaces.reshape((k,) + imshape)
    for i in range(0, k):
        plt.figure()
        current = eigenfaces[i]
        current = np.uint8(scale(current))
        plt.imshow(current, cmap = "gray")


def closest_images():
    closest_num = 3
    faces = []
    #ld_rep is a matrix of the low-dimensional (k x 1) representations of every image in X
    ld_rep = (B[:, 0:k].T @ X).T
    for i in range(0, len(test_names)):
        #read in image, flatten and take away mean face
        iface = cv2.cvtColor(cv2.imread(test_names[i]), cv2.COLOR_RGB2GRAY)
        iface = iface.flatten()
        iface = iface - mean
        #find low-dimensional represenation
        iface_low = B[:, 0:k].T @ iface
        #find distances between low-dimensional rep of test image and training images
        distances = np.linalg.norm(ld_rep - iface_low, axis = 1)
        #return the index of the training image (relative to the list of training images) and the indices of 
        #the three closest training images (relative to the list of training images)
        return_v = np.append(np.array([i]), distances.argsort()[:closest_num])
        faces.append(return_v)

    #display faces
    for i in range(0, len(test_names)):
        plt.figure()
        plt.subplot(221)
        plt.title("Test Image")
        test_face = cv2.imread(test_names[faces[i][0]])
        plt.imshow(test_face)

        plt.subplot(222)
        plt.title("Closest Match")
        plt.imshow(training_images[faces[i][1]], cmap = 'gray')

        plt.subplot(223)
        plt.title("2nd Closest Match")
        plt.imshow(training_images[faces[i][2]], cmap = 'gray')

        plt.subplot(224)
        plt.title("3rd Closest Match")
        plt.imshow(training_images[faces[i][3]], cmap = 'gray')

    return faces
        
#change directory here ############################################
path = os.getcwd()
path_test = "testset"
path_tr = "Yale-FaceA/trainingset"
###################################################################
os.chdir(path_tr)
#load training set
image_names = os.listdir()
image1 = cv2.imread(image_names[75])
imshape = (image1.shape[0], image1.shape[1])

test_names = os.listdir()
X = np.zeros((len(image_names), image1.shape[0] * image1.shape[1]))
training_images = np.zeros((len(image_names),) + imshape)


for i in range(0, len(image_names)):
    current = cv2.imread(image_names[i])
    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    training_images[i] = current
    X[i] = current.flatten()

training_images = np.uint8(training_images)
k = 15
mean = np.mean(X, axis = 0)
X = X - mean
X = X.T
#covariance matrix
S = (1/X.shape[1]) * X.T @ X
evalues, B = np.linalg.eig(S)
#recover eigenvectors
B = X @ B
B = B / np.linalg.norm(B, axis = 0)

#display image
os.chdir("..")
os.chdir(path_test)
test_names = os.listdir()
faces = closest_images()
#eigenfaces()
print("Variance captured in ", k, " principal components was: ",principal_c())
mean_image = mean.reshape(imshape)
mean_image = np.uint8(scale(mean_image))
plt.figure()
plt.imshow(mean_image, cmap = "gray")

plt.show()
cv2.waitKey()