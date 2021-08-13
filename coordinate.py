import cv2
import numpy as np
import pandas as pd
from skimage.io import imread

#image = cv2.imread('1.png')
image = imread('5.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Set threshold level
threshold_level = 50

# Find coordinates of all pixels below threshold
coords = np.column_stack(np.where(gray < threshold_level))
#coords = np.row_stack(np.where(gray < threshold_level))
coords = pd.DataFrame(coords) 
coords.to_csv('data5.csv' , index=False, header=False)
