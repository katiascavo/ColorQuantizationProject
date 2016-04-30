from pylab import imread,imshow,figure,show,subplot
import random
import scipy
from PIL import Image
from numpy import reshape
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans,vq, kmeans2

img = imread('C:/Users/katia/PycharmProjects/FinalVersionProject/testedImages/crop.jpg')

Outfile=open('C:/Users/katia/PycharmProjects/FinalVersionProject/csvDataset/completeDataset.txt', 'w')
Outfile1=open('C:/Users/katia/PycharmProjects/FinalVersionProject/csvDataset/classiColore.txt', 'w')

n_clusters = 5


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

image = np.array(img, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image.shape)
assert d == 3
image_array = np.reshape(image, (w * h, d))


# reshaping the pixels matrix
pixel = reshape(img,(img.shape[0]*img.shape[1],3))

# performing the clustering
centroids,_ = kmeans2(image_array.astype("float"),n_clusters)

# quantization
qnt,_ = vq(image_array,centroids)


compressedImg = recreate_image(centroids, qnt, w, h)
#plotClustered = np.reshape(compressedImg,(w*h,d))
rescaled = (255.0 / compressedImg.max() * (compressedImg - compressedImg.min())).astype(np.uint8)
test = Image.fromarray(rescaled)
#test.save("baboon.jpeg")
pixel1 = reshape(test,(img.shape[0]*img.shape[1],3))


#scrittura su txt/csv
id = 0
cluster = 0

fullList = list()

for j in pixel:
    list1 = list()
    list1 = [j[0], j[1], j[2], qnt[cluster]]
    fullList.append(list1)
    cluster = cluster+1

randomize = reshape(fullList,(img.shape[0]*img.shape[1],4))

#random.shuffle(randomize)
for i in randomize:
    print >>Outfile, "%s;%s;%s;%s;%s" % (id, i[0], i[1], i[2], i[3])
    id = id+1


Outfile.close()

cluster1 = 0
for i in pixel1:
    print >>Outfile1, "%s;%s;%s" % (i[0], i[1], i[2])
    cluster1 = cluster1+1


Outfile1.close()



def kFoldCrossValidation(num_folder):
    #per 5 chunk
    files = ["classe0.csv", "classe1.csv", "classe2.csv", "classe3.csv", "classe4.csv"]

    classe0 = 0
    classe1 = 0
    classe2 = 0
    classe3 = 0
    classe4 = 0
    list0 = list()
    list1 = list()
    list2 = list()
    list3 = list()
    list4 = list()
    for i in randomize:
        if i[3] == 0:
            classe0 = classe0 + 1
            list0.append(i)
        elif i[3] == 1:
            classe1 = classe1 + 1
            list1.append(i)
        elif i[3] == 2:
            classe2 = classe2 + 1
            list2.append(i)
        elif i[3] == 3:
            classe3 = classe3 + 1
            list3.append(i)
        else:
            classe4 = classe4 + 1
            list4.append(i)


    print "classe 0", classe0
    print "classe 1", classe1
    print "classe 2", classe2
    print "classe 3", classe3
    print "classe 4", classe4

    print

    istances0 = (classe0) / (num_folder)
    print "per fold",istances0
    istances1 = (classe1) / (num_folder)
    print "per fold",istances1
    istances2 = (classe2) / (num_folder)
    print "per fold",istances2
    istances3 = (classe3) / (num_folder)
    print "per fold",istances3
    istances4 = (classe4) / (num_folder)
    print "per fold",istances4

    output = open('C:/Users/katia/PycharmProjects/FinalVersionProject/csvDataset/' + files[0], 'w')
    for i in list0:
        print >> output, "%s;%s;%s;%s" % (i[0], i[1], i[2], i[3])

    output.close()

    output = open('C:/Users/katia/PycharmProjects/FinalVersionProject/csvDataset/' + files[1], 'w')
    for i in list1:
        print >> output, "%s;%s;%s;%s" % (i[0], i[1], i[2], i[3])

    output.close()

    output = open('C:/Users/katia/PycharmProjects/FinalVersionProject/csvDataset/' + files[2], 'w')
    for i in list2:
        print >> output, "%s;%s;%s;%s" % (i[0], i[1], i[2], i[3])

    output.close()

    output = open('C:/Users/katia/PycharmProjects/FinalVersionProjectn/csvDataset/' + files[3], 'w')
    for i in list3:
        print >> output, "%s;%s;%s;%s" % (i[0], i[1], i[2], i[3])

    output.close()

    output = open('C:/Users/katia/PycharmProjects/FinalVersionProject/csvDataset/' + files[4], 'w')
    for i in list4:
        print >> output, "%s;%s;%s;%s" % (i[0], i[1], i[2], i[3])

    output.close()

kFoldCrossValidation(10)


###############################################################
#                MSE-PSNR evaluating measures for K-means
###############################################################
# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
    # return the MSE, the lower the error, the more "similar"
	# the two images are
err = np.sum((img.astype("float") - compressedImg.astype("float")) ** 2)
MSE = err /float(img.shape[0]*img.shape[1])
print "MSE: ",MSE
########## PSNR
psnr = 20 * math.log(255/ math.sqrt(MSE), 10)
print "PSNR: ",psnr
print


#plot results
figure(1)
subplot(211)
imshow(img)

figure(2)
subplot(111)
imshow(compressedImg)

show()

# visualizing the centroids into the RGB space
from mpl_toolkits.mplot3d import Axes3D
#ax = fig.gca(projection='3d')
fig = plt.figure(2, figsize=(8, 7))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c=centroids.astype(np.float),s=250)
'''
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
'''

plt.show()


'''
#ax = fig.gca(projection='3d')
fig = plt.figure(3, figsize=(8, 7))
plt.clf()
ax = Axes3D(fig)
plt.cla()
ax.scatter(image_array[:,0],image_array[:,1],image_array[:,2],c=image_array.astype(np.float),s=65)
ax.patch.set_facecolor('black')
#ax.patch.set_color('')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
'''

'''
img1 = imread('C:/Users/katia/PycharmProjects/ImageSegmentation/colorQuantization/baboon.jpeg')
image1 = np.array(img1, dtype=np.float64) / 255

# Load Image and transform to a 2D numpy array.
w, h, d = original_shape = tuple(image1.shape)
assert d == 3
image_array11 = np.reshape(image1, (w * h, d))


#ax = fig.gca(projection='3d')
fig = plt.figure(3, figsize=(8, 7))
plt.clf()
ax = Axes3D(fig)#rect=[0, 0, .95, 1], elev=48, azim=134
plt.cla()
ax.scatter(image_array11[:,0],image_array11[:,1], image_array11[:,2],c=image_array11.astype(np.float) ,s=65)
ax.patch.set_facecolor('black')


ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
'''