# Pupil Detector

## Gradient Intersect

The PupilDetector module contains the GradientIntersect class, which is an implementation of the "Accurate Eye Centre Localisation by Means of Gradients" paper by Fabian Timm and Erhardt Barth.
This method uses the maximum intersection of gradients as an estimation of the center of a pupil.

![alt text](https://raw.githubusercontent.com/jonnedtc/PupilDetector/master/pupil.png "Example")


## Usage

Import the GradientIntersect class from the PupilDetector module.

Transform your image to a grayscale numpy ndarray, with shape: (height, width).

Create a new GradientIntersect object.

Call the locate method on the object with the grayscale image as an argument, this returns the (y, x) position of the pupil.

When analyzing a video you can use the track method, to search around the pupil location of the previous frame.

Call the track method with the following arguments: the grayscale image and a tuple (y, x) with the previous location.

## Speed

Increase speed of locate by increasing the accuracy variable.

The algorithm takes a sample ones every accuracy*accuracy pixels, after this it takes a closer look around the best sample.

Increase speed of track by decreasing radius and distance.

Radius is half the width and height around the previous location (py, px), the whole pupil should still be visible in this area.

Distance is the maximum amount of pixels between the new and old location.

## Examples

```
import numpy as np
from matplotlib import pyplot as plt
from PupilDetector import GradientIntersect

FILENAME = "image.png"
frame = plt.imread(FILENAME)
gray = np.sum(frame,axis=2)

gi = GradientIntersect()
print gi.locate(gray)
```

```
import cv2
import numpy as np
from PupilDetector import GradientIntersect

FILENAME = "movie.webm"
cap = cv2.VideoCapture(FILENAME)
ret,frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

gi = GradientIntersect()
loc = gi.locate(gray)
print loc

ret,frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

loc = gi.track(gray, loc)
print loc
```
