import csv
import cv2

import numpy as np
import tensorflow as tf
import keras


rows = []
with open('./sim/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rows.append(row)

print(rows[0][0])

images = []
measurements = []

for row in rows:
    source_path = row[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    print('test')
    local_path = './sim/'+filename
    print(local_path)
    #Image processing
    image = cv2.imread(local_path)
    images.append(image)
    measurement = row[3]
    measurements.append(measurement)

print(len(images))
print(len(measurements))

exit()
