# Pupil Detector

The PupilDetector module contains the GradientIntersect class, which is an implementation of the "Accurate Eye Centre Localisation by Means of Gradients" paper by Fabian Timm and Erhardt Barth.
This method uses the maximum intersection of gradients as an estimation of the center of a pupil.

# Usage

Import the GradientIntersect class from the PupilDetector module.

Transform your image to a grayscale numpy ndarray, with shape: (height, width).

Create a new GradientIntersect object with at least the shape of the ndarray as argument.

Call the locate method on the object with the ndarray as argument, this returns the (y, x) position of the pupil.

As long as the images are the same size you can keep calling the locate method with a new image.

# Speed

To speed up the method add the threshold and step arguments when creating the GradientIntersect object.

The threshold argument is the percentile of brighter pixels to ignore when searching for the pupil.

The step argument is the amount of pixels to skip when searching for the pupil.
This takes a sample every few pixels and does a more thorough search close to the maximum sample.

# Examples

'''
import numpy as np
from matplotlib import pyplot as plt
from EyeLocation import GradientIntersect

FILENAME = "image.png"
frame = plt.imread(FILENAME)
gray = np.sum(frame,axis=2)

mg = GradientIntersect(gray.shape, threshold = 50, step = 8)
print mg.locate(gray)
'''

'''
import cv2
import numpy as np
from EyeLocation import GradientIntersect

FILENAME = "movie.webm"
cap = cv2.VideoCapture(FILENAME)
ret,frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mg = GradientIntersect(gray.shape, threshold = 50, step = 8)
print mg.locate(gray)
'''