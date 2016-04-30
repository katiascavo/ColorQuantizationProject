# resize an image using the PIL image library
# free from:  http://www.pythonware.com/products/pil/index.htm
# tested with Python24        vegaseat     11oct2005

import numpy as np
import scipy.cluster
from PIL import Image
from numpy import *
from numpy import reshape
from pylab import imshow,figure,show,subplot,savefig
from scipy.cluster.vq import vq

from clustering_algorithms.kmeans import FuzzyKMeans, KMedians

# open an image file (.bmp,.jpg,.png,.gif) you have in the working folder
imageFile = "C:/Users/katia/PycharmProjects/FinalVersionProject/testedImages/4.2.03.jpg"
im1 = Image.open(imageFile)
width, height = im1.size[0], im1.size[1]
# adjust width and height to your needs
width2 = width/2
height2 = height/2

NUM_CLUSTER = 5
# use one of these filter options to resize the image

#im3 = im1.resize((width2, height2), Image.BILINEAR)   # linear interpolation in a 2x2 environment
im3= im1 #è stata tolta la compressione

#ext = ".jpg"
#im3.save("BILINEAR" + ext)
#per la visualizzazione dell'immagine
pixel1 = reshape(im3,(im3.size[0],im3.size[1],3))

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image



prova = np.array(pixel1, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(prova.shape)
assert d == 3
image_array = np.reshape(prova, (w * h, d))

########################################################################
#                                  K-Means
########################################################################

# performing the clustering
centroids,_ = scipy.cluster.vq.kmeans(image_array.astype(float), NUM_CLUSTER)
# quantization
qnt,_ = vq(image_array,centroids)

clustered = recreate_image(centroids, qnt, width, height)

###############################################################
#                MSE-PSNR evaluating measures for K-means
###############################################################
# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    # return the MSE, the lower the error, the more "similar"
	# the two images are
err = np.sum((pixel1.astype("float") - clustered.astype("float")) ** 2)
mse = err /float(pixel1.shape[0] * pixel1.shape[1])
print "MSE k-means: ",mse
########## PSNR
psnr = 20 * math.log(255/ math.sqrt(mse), 10)
print "PSNR k-means: ",psnr
print


#########################################################################
#                               K-Means 2
#########################################################################

centroids2,_ = scipy.cluster.vq.kmeans2(image_array.astype(float), NUM_CLUSTER)
# quantization
qnt2,_ = vq(image_array,centroids2)

clustered2 = recreate_image(centroids2, qnt2, width, height)
###############################################################
#                MSE-PSNR evaluating measures for K-means2
###############################################################
# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    # return the MSE, the lower the error, the more "similar"
	# the two images are
err = np.sum((pixel1.astype("float") - clustered2.astype("float")) ** 2)
mse = err /float(pixel1.shape[0] * pixel1.shape[1])
print "MSE k-means2: ",mse
########## PSNR
psnr = 20 * math.log(255/ math.sqrt(mse), 10)
print "PSNR k-means2: ",psnr
print

##########################################################################
#                                 Fuzzy C-Means
##########################################################################
# Notice that the starting values for the memberships could be randomly choosen,
# at least for simple cases like this. You could try the lines below to
# initialize the membership array:
#

fcm = FuzzyKMeans(k=NUM_CLUSTER, m=2)
fcm.fit(image_array)
fkmImage = recreate_image(fcm.cluster_centers_, fcm.labels_, width, height)


###############################################################
#             MSE-PSNR evaluating measures for Fuzzy C-Means
###############################################################
# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    # return the MSE, the lower the error, the more "similar"
	# the two images are
err = np.sum((pixel1.astype("float") - fkmImage.astype("float")) ** 2)
mse = err /float(pixel1.shape[0] * pixel1.shape[1])
print "MSE Fuzzy C-means: ",mse
########## PSNR
psnr = 20 * math.log(255/ math.sqrt(mse), 10)
print "PSNR Fuzzy C-means: ",psnr
print

##########################################################################
#                                K-medoids
##########################################################################


kmed = KMedians(k=NUM_CLUSTER)
kmed.fit(image_array)
kmedImage = recreate_image(kmed.cluster_centers_, kmed.labels_, width, height)


###############################################################
#             MSE-PSNR evaluating measures for Fuzzy C-Means
###############################################################
# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    # return the MSE, the lower the error, the more "similar"
	# the two images are
err = np.sum((pixel1.astype("float") - kmedImage.astype("float")) ** 2)
mse = err /float(pixel1.shape[0] * pixel1.shape[1])
print "MSE k-medians: ",mse
########## PSNR
psnr = 20 * math.log(255/ math.sqrt(mse), 10)
print "PSNR k-medians: ",psnr

'''
#compressed
figure(1)
subplot(111)
test = imshow(pixel1)
'''
#k-means
figure(2)
subplot(111)
imshow(clustered)

#k-means2
figure(3)
subplot(111)
imshow(clustered2)

#fuzzy k-means
figure(4)
subplot(111)
imshow(fkmImage)
#k-medians
figure(5)
subplot(111)
imshow(kmedImage)
show()
