# Eye Centre Localisation using Means of Gradients
# Implementation of paper by Fabian Timm and Erhardt Barth
# "Accurate Eye Centre Localisation by Means of Gradients"

# Author: Jonne Engelberts
# Increase speed by setting threshold lower and step higher.
# Threshold is the percentile of bright area included.
# Step is the amount of pixels initially skipped.

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

class GradientIntersect:

	def __init__(self, (height, width), threshold=100, step=1):

		self.height = height
		self.width = width
		self.grid = self.createGrid(height, width)
		self.area = threshold
		self.accuracy = step
		if self.accuracy < 1:
			self.accuracy = 1

	def createGrid(self, Y, X):
    
		# create grid vectors
		grid = np.array([[y,x] for y in range(1-Y,Y) for x in range (1-X,X)], dtype='float')

		# normalize grid vectors
		grid2 = np.sqrt(np.einsum('ij,ij->i',grid,grid))
		grid2[grid2==0] = 1
		grid /= grid2[:,np.newaxis]

		# reshape grid for easier displacement selection
		grid = grid.reshape(Y*2-1,X*2-1,2)
		
		# return values
		return grid

	def locate(self, image):
		
		# get image size
		Y, X = image.shape
		
		# create empty score matrix
		scores = np.zeros((Y,X))

		# normalize image
		image = (image.astype('float') - np.min(image)) / np.max(image)

		# blur image
		blurred = gaussian_filter(image, sigma=5)
		retain = np.percentile(blurred, self.area)

		# calculate gradients
		gy, gx = np.gradient(image)
		gx = np.reshape(gx, (X*Y,1))
		gy = np.reshape(gy, (X*Y,1))
		gradient = np.hstack((gy,gx))

		# normalize gradients
		gradient2 = np.sqrt(np.einsum('ij,ij->i',gradient,gradient))
		gradient2[gradient2==0] = 1
		gradient /= gradient2[:,np.newaxis]

		# loop through all pixels
		for cy in range(0,Y,self.accuracy):
			for cx in range(0,X,self.accuracy):

				if image[cy,cx] < retain:

					# select displacement
					displacement = self.grid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
					displacement = np.reshape(displacement,(X*Y,2))

					# calculate score
					score = np.einsum('ij,ij->i', displacement, gradient)
					score = np.einsum('i,i->', score, score)
					scores[cy,cx] = score

		# multiply with the blurred darkness
		scores = scores * (1-blurred)

		# if we skipped pixels, get more accurate around the best pixel
		if self.accuracy>1:

			# get maximum value index 
			(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)

			# prevent maximum value index from being close to 0 or max
			yval = min(max(yval,self.accuracy), Y-self.accuracy-1)
			xval = min(max(xval,self.accuracy), X-self.accuracy-1)

			# loop through new pixels
			for cy in range(yval-self.accuracy,yval+self.accuracy+1):
				for cx in range(xval-self.accuracy,xval+self.accuracy+1):

					# select displacement
					displacement = self.grid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
					displacement = np.reshape(displacement,(X*Y,2))

					# calculate score
					score = np.einsum('ij,ij->i', displacement, gradient)
					score = np.einsum('i,i->', score, score)
					scores[cy,cx] = score * (1-blurred[cy,cx])

		# get maximum value index 
		(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)
		
		# return values
		return (yval, xval)

	def locatePartial(self, image, (py, px), radius):
		
		# make sure py, px is within image radius
		px = min(max(px,0),self.width)
		py = min(max(py,0),self.height)

		# radius can only be half of the image at max
		if radius*2 + 1 > self.width:
			radius = int(self.width / 2) - 1
			print("PupilDetector: resized radius to: " + str(radius))
		if radius*2 + 1 > self.height:
			radius = int(self.height / 2) - 1
			print("PupilDetector: resized radius to: " + str(radius))

		# make sure the new box is within image radius
		if px - radius < 0:
			px = radius
		if px + radius > self.width:
			px = self.width - radius
		if py - radius < 0:
			py = radius
		if py + radius > self.height:
			py = self.height - radius

		# get partial image
		image = image[py-radius:py+radius+1,px-radius:px+radius+1]

		# get partial grid
		ydif = self.height - radius*2 - 1
		xdif = self.width - radius*2 - 1
		partialGrid = self.grid[0+ydif:self.height*2-1-ydif, 0+xdif:self.width*2-1-ydif]

		# get image size
		Y, X = image.shape
		
		# create empty score matrix
		scores = np.zeros((Y,X))

		# normalize image
		image = (image.astype('float') - np.min(image)) / np.max(image)

		# blur image
		blurred = gaussian_filter(image, sigma=5)
		retain = np.percentile(blurred, self.area)

		# calculate gradients
		gy, gx = np.gradient(image)
		gx = np.reshape(gx, (X*Y,1))
		gy = np.reshape(gy, (X*Y,1))
		gradient = np.hstack((gy,gx))

		# normalize gradients
		gradient2 = np.sqrt(np.einsum('ij,ij->i',gradient,gradient))
		gradient2[gradient2==0] = 1
		gradient /= gradient2[:,np.newaxis]

		# loop through all pixels
		for cy in range(0,Y,self.accuracy):
			for cx in range(0,X,self.accuracy):

				if image[cy,cx] < retain:

					# select displacement
					displacement = partialGrid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
					displacement = np.reshape(displacement,(X*Y,2))

					# calculate score
					score = np.einsum('ij,ij->i', displacement, gradient)
					score = np.einsum('i,i->', score, score)
					scores[cy,cx] = score

		# multiply with the blurred darkness
		scores = scores * (1-blurred)

		# if we skipped pixels, get more accurate around the best pixel
		if self.accuracy>1:

			# get maximum value index 
			(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)

			# prevent maximum value index from being close to 0 or max
			yval = min(max(yval,self.accuracy), Y-self.accuracy-1)
			xval = min(max(xval,self.accuracy), X-self.accuracy-1)

			# loop through new pixels
			for cy in range(yval-self.accuracy,yval+self.accuracy+1):
				for cx in range(xval-self.accuracy,xval+self.accuracy+1):

					# select displacement
					displacement = partialGrid[Y-cy-1:Y*2-cy-1,X-cx-1:X*2-cx-1,:]
					displacement = np.reshape(displacement,(X*Y,2))

					# calculate score
					score = np.einsum('ij,ij->i', displacement, gradient)
					score = np.einsum('i,i->', score, score)
					scores[cy,cx] = score * (1-blurred[cy,cx])

		# get maximum value index 
		(yval,xval) = np.unravel_index(np.argmax(scores),scores.shape)
		
		# return values
		return (py+yval-radius, px+xval-radius)

class IsophoteCurvature:

	def __init__(self, blur = 3, minrad = 2, maxrad = 20):

		self.blur = blur
		self.minrad = minrad
		self.maxrad = maxrad

	def locate(self, image):

		# normalize image
		image = gaussian_filter(image, sigma=self.blur)
		image = (image.astype('float') - np.min(image))
		image = image / np.max(image)
		
		# calculate gradients
		Ly, Lx = np.gradient(image)
		Lyy, Lyx = np.gradient(Ly)
		Lxy, Lxx = np.gradient(Lx)
		Lvv = Ly**2 * Lxx - 2*Lx * Lxy * Ly + Lx**2 * Lyy
		Lw =  Lx**2 + Ly**2
		Lw[Lw==0] = 0.001
		Lvv[Lvv==0] = 0.001
		k = - Lvv / (Lw**1.5)
		
		# calculate displacement
		Dx =  -Lx * (Lw / Lvv)
		Dy =  -Ly * (Lw / Lvv)
		displacement = np.sqrt(Dx**2 + Dy**2)
		
		# calculate curvedness
		curvedness = np.absolute(np.sqrt(Lxx**2 + 2 * Lxy**2 + Lyy**2))
		center_map = np.zeros(image.shape, image.dtype)
		(height, width)=center_map.shape   
		for y in range(height):
			for x in range(width):
				if Dx[y][x] == 0 and Dy[y][x] == 0:
					continue
				if (x + Dx[y][x])>0 and (y + Dy[y][x])>0:
					if (x + Dx[y][x]) < center_map.shape[1] and (y + Dy[y][x]) < center_map.shape[0] and k[y][x]<0:
						if displacement[y][x] >= self.minrad and displacement[y][x] <= self.maxrad:
							center_map[int(y+Dy[y][x])][int(x+Dx[y][x])] += curvedness[y][x]
		center_map = gaussian_filter(center_map, sigma=self.blur)
		blurred = gaussian_filter(image, sigma=self.blur)
		center_map = center_map * (1-blurred)
		
		# return maximum location in center_map
		position = np.unravel_index(np.argmax(center_map), center_map.shape)
		return position