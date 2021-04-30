import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

obj = cv2.imread("object.jpg")
objGray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)
objBlur = cv2.GaussianBlur(obj, (5, 5), 0)

totalMatchList = []
for fileName in os.listdir("Data"):
    imgScene = cv2.imread("Data/" + fileName)
    imgSceneGray = cv2.cvtColor(imgScene, cv2.COLOR_BGR2GRAY)
    imgSceneBlur = cv2.GaussianBlur(imgSceneGray, (5, 5), 0)
    
    surfObj = cv2.xfeatures2d.SURF_create()
    keyObj, desObj = surfObj.detectAndCompute(objGray, None)
    keyScene, desScene = surfObj.detectAndCompute(imgSceneBlur, None)

    indexParam = dict(algorithm=0)  
    searchParam = dict(checks=50)

    flannObj = cv2.FlannBasedMatcher(indexParam, searchParam)
    matches = flannObj.knnMatch(desObj, desScene, 2)

    matchesMask = []

    for i in range(len(matches)):
        matchesMask.append([0, 0])

    totalMatch = 0

    for index, (first_best, second_best) in enumerate(matches):
        if first_best.distance < 0.7 * second_best.distance:
            matchesMask[index] = [1, 0]
            totalMatch += 1
    totalMatchList.append(totalMatch)        
            
maxValue = max(totalMatchList)
maxIndex = totalMatchList.index(maxValue)

imgScene = cv2.imread("Data/"+ str(maxIndex)+".jpg")
imgSceneGray = cv2.cvtColor(imgScene, cv2.COLOR_BGR2GRAY)
imgSceneBlur = cv2.GaussianBlur(imgSceneGray, (5, 5), 0)
    
surfObj = cv2.xfeatures2d.SURF_create()
keyObj, desObj = surfObj.detectAndCompute(objGray, None)
keyScene, desScene = surfObj.detectAndCompute(imgSceneBlur, None)

indexParam = dict(algorithm=0)  
searchParam = dict(checks=50)

flannObj = cv2.FlannBasedMatcher(indexParam, searchParam)
matches = flannObj.knnMatch(desObj, desScene, 2)

matchesMask = []

for i in range(len(matches)):
    matchesMask.append([0, 0])

totalMatch = 0

for index, (first_best, second_best) in enumerate(matches):
    if first_best.distance < 0.7 * second_best.distance:
        matchesMask[index] = [1, 0]
        totalMatch += 1
        
final_img = cv2.drawMatchesKnn(
    obj,
    keyObj,
    imgScene,
    keyScene,
    matches,
    None,
    matchColor=[0, 255, 0],
    singlePointColor=[255, 0, 0],
    matchesMask=matchesMask
    )
    
plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.show()
    
    


