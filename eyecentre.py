# Eye Centre Localisation using Means of Gradients
# Implementation of paper by Fabian Timm and Erhardt Barth
# "Accurate Eye Centre Localisation by Means of Gradients"

# Author: Jonne Engelberts

import numpy as np

class EyeCentre:

	def __init__(self, image):

		self.image = image

