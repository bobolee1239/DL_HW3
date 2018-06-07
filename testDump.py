#!/usr/bin/env python3

import joblib
from glob import glob
import os 
import cv2

fileDir = './DL_HW3/faces/'

files = glob(os.path.join(fileDir, '*.jpg'))

imgs = []
for f in files:
	img = cv2.imread(f, cv2.IMREAD_COLOR)
	img = img[:, :, ::-1]
	img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
	img = img / 255.0

	imgs.append(img)

joblib.dump(imgs, "./save1.pkl")

