import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

#list file names of frame
frames = os.listdir("Frames/")


#Read frames
images = []
for i in frames:
    img = cv2.imread("Frames/" + i)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (25, 25), 0)
    images.append(img)

images = np.array(images)

nonzero = []

for i in range(len(images) - 1):

    mask = cv2.absdiff(images[i], images[i + 1])
    _ , mask = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY)

    #cv2.imshow("Mask", mask)

    num = np.count_nonzero(mask.ravel())
    nonzero.append(num)

x = np.arange(0, len(images) - 1)
y = nonzero

#plt.figure(figsize=(20,4))
#plt.scatter(x,y)
#plt.show()

#Prepare and segment one frame

img = cv2.imread("Frames/69.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (25, 25), 0)
plt.figure(figsize=(5,10))
plt.imshow(gray, cmap='gray')
plt.show()


#Create mask
_, mask = cv2.threshold(gray, 200, 245, cv2.THRESH_BINARY)
plt.figure(figsize=(5,5))
plt.imshow(mask,cmap='gray')
plt.show()


#Find contours

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_copy = np.copy(gray)

cv2.drawContours(img_copy, contours, -1, (0, 255, 0), 3)

plt.figure(figsize=(5,5))
plt.imshow(img_copy, cmap='gray')
plt.show()


#Extract the patches from an image using the contours

#!rm -r patch/*

num = 20
cnt = 0

for i in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[i])

    numer = min([w, h])
    denom = max([w, h])
    ratio = numer / denom

    if (x >= num and y >= num):
        xmin, ymin = x-num, y-num
        xmax, ymax = x+w+num, y+h+num
    else:
        xmin, ymin = x, y
        xmax, ymax = x+w, y+h

    if (ratio >= 0.5 and ((15<=w<=30) and (15<=h<=30)) ):
        print(cnt, x, y, w, h, ratio)
        cv2.imwrite("Patch/" + str(cnt) + ".png", img[ymin:ymax, xmin:xmax])
        cnt += 1



#P2:

import pandas as pd

folders = os.listdir("Data/")

images = []
labels = []


#Original:
for folder in folders:
    files = os.listdir("Data/" + folder)
    for file in files:
        img = cv2.imread("Data/" + folder + "/" + file, 0)
        img = cv2.resize(img, (25, 25))

        images.append(img)

        labels.append(int(folder))

#Modified:
"""
files = os.listdir("Data/")
for file in files:
    img = cv2.imread("Data/" + file, 0)
    try:
        img = cv2.resize(img, (25, 25))

        images.append(img)
        labels.append(int(file))
    except:
        print('Something is not right')

"""



images = np.array(images)
features = images.reshape(len(images), -1)

#Split dataset into training and validation

from sklearn.model_selection import train_test_split

x_tr, x_val, y_tr, y_val = train_test_split(features, labels, test_size=0.2, stratify= labels, random_state=0)

#Build baseline model for identifying the patch containing ball

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(max_depth=3)
rfc.fit(x_tr, y_tr)

#Evaluate model based on validation data

from sklearn.metrics import classification_report

y_pred = rfc.predict(x_val)
print(classification_report(y_val, y_pred))

#Repeat similar steps for each frame in a video followed by classification

ball_df = pd.DataFrame(columns=['frame','x','y','w','h'])

for idx in range(len(frames)):

    img= cv2.imread('Frames/' + frames[idx])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (25, 25), 0)
    _ , mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num = 20
    cnt = 0
    df = pd.DataFrame(columns=['frame', 'x', 'y', 'w', 'h'])
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])

        numer = min([w, h])
        denom = max([w, h])
        ratio = numer / denom

        if (x >= num and y >= num):
            xmin, ymin = x-num, y-num
            xmax, ymax = x+w+num, y+h+num
        else:
            xmin, ymin = x, y
            xmax, ymax = x+w, y+h

        if (ratio >= 0.5):
            df.loc[cnt, 'frame'] = frames[idx]
            df.loc[cnt, 'x'] = x
            df.loc[cnt, 'y'] = y
            df.loc[cnt, 'w'] = w
            df.loc[cnt, 'h'] = h

            cv2.imwrite("Patch/" + str(cnt) + ".png", img[ymin:ymax, xmin:xmax])
            cnt += 1

    files = os.listdir("Patch/")
    if (len(files) > 0):

        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        test = []

        for file in files:
            img = cv2.imread("Patch/" + file, 0)
            img = cv2.resize(img, (25, 25))
            test.append(img)

        test = np.array(test)

        test = test.reshape(len(test), -1)
        y_pred = rfc.predict(test)
        prob = rfc.predict_proba(test)

        if 0 in y_pred:
            ind = np.where(y_pred==0)[0]
            proba = prob[:, 0]
            confidence = proba[ind]
            confidence = [i for i in confidence if i>0.7]
            
            if (len(confidence) > 0):

                maximum = max(confidence)
                ball_file = files[list(proba).index(maximum)]

                img = cv2.imread("Patch/" + ball_file)
                cv2.imwrite("Ball/" + str(frames[idx]), img)

                no = int(ball_file.split(".")[0])
                ball_df.loc[idx] = df.loc[no]
            else:
                ball_df.loc[idx, 'frame'] = frames[idx]

        else:
            ball_df.loc[idx, 'frame'] = frames[idx]


ball_df.dropna(inplace=True)
print(ball_df)
