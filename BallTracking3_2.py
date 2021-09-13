import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import re

#list file names of frame
frames = os.listdir("Frames/")
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

print("Loading Frames:")
print(frames)
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
    _ , mask = cv2.threshold(mask, 70, 255, cv2.THRESH_BINARY)

    num = np.count_nonzero(mask.ravel())
    nonzero.append(num)

x = np.arange(0, len(images) - 1)
y = nonzero

plt.figure(figsize=(20,4))
plt.scatter(x,y)
plt.show()

#Prepare and segment one frame

img = cv2.imread("Frames/70.png")
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

    if (ratio >= 0.5 and ((20<=w<=30) and (20<=h<=30)) ):
        print(cnt, x, y, w, h, ratio)
        cv2.imwrite("Patch/" + str(cnt) + ".png", img[ymin:ymax, xmin:xmax])
        cnt += 1



#NEW:

#Draw Rectangle
frames = os.listdir("Frames/")
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

print(frames)

frame_array = []
frame_counter = 0

#Read frames
for i in frames:
    print("Processing", i)
    print("Frame:", frame_counter)
    img = cv2.imread("Frames/" + i)

    frame_array.append(img)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (25, 25), 0)

    _, mask = cv2.threshold(img_gray, 200, 245, cv2.THRESH_BINARY)

    if (frame_counter > 0):
        img2 = frame_array[frame_counter - 1]
        #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.GaussianBlur(img2, (25, 25), 0)
        #img2mask = cv2.threshold(img2, 200, 245, cv2.THRESH_BINARY)
        diff = cv2.absdiff(img, img2)
        diffMask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        #_, diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        #img_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diffMask = cv2.GaussianBlur(diffMask, (25, 25), 0)

        th = 1
        idiffMask = diffMask > th
        
        canvas = np.zeros_like(img, np.uint8)
        canvas[idiffMask] = img[idiffMask]

        #See if this works. If not, omit:
        cv2.imwrite("TEST/diff" + i, canvas)

        mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 200, 245, cv2.THRESH_BINARY)
    #else:
        #img_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_copy = np.copy(img_gray)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    num = 20
    cnt = 0

    for j in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[j])

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
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cnt += 1

    cv2.imwrite("NewFrames/" + i, img)

    frame_counter += 1



frames = os.listdir("NewFrames/")
frames.sort(key=lambda f: int(re.sub('\D', '', f)))

frame_array = []

for i in range(len(frames)):
    #Reading each frame
    img = cv2.imread("NewFrames/" + frames[i])
    height, width, layers = img.shape
    size = (width, height)
    #Inserting frame to array
    frame_array.append(img)


out = cv2.VideoWriter('NewVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, size)

for i in range(len(frame_array)):
    #Write to image array
    out.write(frame_array[i])


out.release()
cv2.destroyAllWindows()

