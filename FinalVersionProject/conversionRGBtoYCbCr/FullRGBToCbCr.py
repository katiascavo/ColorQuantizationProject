#====================================================================================================
# File        : RGB2YCbCr.py
# Author      : Ghanshyam
# Dated       : 08.10.2010
# Description : The file converts the Windows style BMP image to YCbCr binary stream
#               with a OuputFile.YCbCr extention
#               The script shall be run with the argument as the BMP image. The output file
#               shall be stored in the current directory where the script is executed.
#
# Rev         : v1.0
# ===================================================================================================
# Change Log  :
# ===================================================================================================
# Author      Date        Rev     Description
# Shyam   08.10.2010      v1.0    Created initial source code.
#====================================================================================================
# Copyright(c) 2010:
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU
# General Public License as published by the Free Software Foundation version 3 of the License.
#
# This program is distributed in the hope that it will be useful,but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details. You should have received a copy of the GNU General Public
# License along with this program.If not, see <http://www.gnu.org/licenses/>.
# ===================================================================================================


# import the python imaging library : LIBRARIES IMPORT
#!/usr/bin/python
import sys, time
from PIL import Image
from pylab import imread,imshow,figure,show,subplot
from numpy import reshape
from scipy.cluster.vq import kmeans,vq
import scipy.cluster
import os, math


# open the bmp image  : OPEN AND CONSUME THE BMP FILE
try:
    image= Image.open("C:/Users/katia/PycharmProjects/FinalVersionProject/testedImages/4.2.03.jpg")
    print "Opening jpg image ..............................................................[O.K]"
except:
    print "Error opening. Check if file exists....................[EXIT]"
    sys.exit()

# PARSE FOR SIZE:
print "Input tiff Image size:"
print image.size
width, height = image.size
halfWidth = width/2
halfHeight = height/2
#preleva i pixel formato RGB
pixel=list(image.getdata())

Outfile=open('C:/Users/katia/PycharmProjects/FinalVersionProject/conversionRGBtoYCbCr/completeDatasetRGB.txt', 'w')
#dimensione dell'immagine
Size = image.size[0]*image.size[1]
print Size

listaValori =list()

# Convert to YCbCr
for i in pixel:

    ## Y Equation
    Y = (0.299 * i[0]) + (0.587 * i[1]) + (0.114 * i[2])+ 0
    ##Cb Equation
    Cb = 128 - (0.168736 * i[0]) - (0.331264 * i[1]) + (0.5 * i[2])
    ## Cr Equantion
    Cr = 128 + (0.5 * i[0]) - (0.418688 * i[1]) - (0.081312 * i[2])

    valori = [Y, Cb, Cr]

    listaValori.append(valori)


#converasione formato 4:2:2
colorValue = list()
i = 0
while(i < len(listaValori)):
    colorx = [listaValori[i][0],listaValori[i+1][0],listaValori[i][1],listaValori[i][2]]
    colorValue.append(colorx)
    i = i+2

print len(colorValue)

#conversione formato 4:2:0
colorValue2 = list()
z = 0
j = halfWidth #256 se 512
while( z < len(colorValue)):
    for x in xrange(halfWidth):
        matrix = [colorValue[z][0],colorValue[z][1],colorValue[j][0],colorValue[j][1],colorValue[z][2], colorValue[z][3]]
        colorValue2.append(matrix)
        z = z+1
        j = j+1
    j = j + halfWidth
    z = z + halfWidth

print len(colorValue2)

#####################################################
# calcolo da YCbCr a RGB
# media delle Y
#####################################################
x = 0
fullRGB = list() #lista dei valori rgb dell'immagine compressa
while(x < len(colorValue2)):
    yAverage = (colorValue2[x][0]+colorValue2[x][1]+colorValue2[x][2]+colorValue2[x][3])/4
    r = math.trunc(round(1.402*((colorValue2[x][5])-128) + yAverage));
    g = math.trunc(round(-0.34414*((colorValue2[x][4])-128)-0.71414*((colorValue2[x][5])-128) + yAverage));
    b = math.trunc(round(1.772*((colorValue2[x][4])-128) + yAverage));
    rgb = [r,g,b]
    fullRGB.append(rgb)
    x = x+1

#to plot compressed image
compressedImage = reshape(fullRGB,(halfWidth,halfHeight, 3))

##################################################################################################
####                           clustering of image                                            ####
##################################################################################################
pixel1 = reshape(fullRGB,(halfWidth*halfHeight,3))
# performing the clustering
centroids,_ = scipy.cluster.vq.kmeans(pixel1.astype(float), 5)
# quantization
qnt,_ = vq(pixel1,centroids)

# reshaping the result of the quantization
centers_idx = reshape(qnt,(halfWidth, halfHeight))
clustered = centroids[centers_idx]

'''
figura 1: immagine originale e immagine compressa
figura 2: immagine clusterizzata con k = 5
'''

figure(1)
subplot(111)
imshow(image)
figure(2)
subplot(111)
imshow(compressedImage)
figure(3)
subplot(111)
imshow(clustered)
show()

'''
#scrittura su txt/csv
cluster = 0
for i in fullRGB:
    print >>Outfile, "%s;%s;%s;%s" % (i[0], i[1], i[2], qnt[cluster])
    cluster = cluster+1


Outfile.close()
'''