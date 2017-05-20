import os
import sys

testimgfolderpath = "/Users/bjartesjursen/Desktop/imagenet/imagenet56x56_release/test"
labelFile = "/Users/bjartesjursen/Desktop/imagenet/imagenet56x56_release/train/train_labels.csv"

images = os.listdir(testimgfolderpath)

trainingFile = open(labelFile,'r')

linesToAdd = []
linesToRemove = []

for line in trainingFile:
    lineElements = line.split(',')
    filename = lineElements[0] + ".JPEG"
    label = lineElements[1].split('\n')
    if filename in images:
        linesToRemove.append(line)
    else:
        linesToAdd.append(line)

trainingFile.close()

trainingFile = open("train_labels.txt",'w')

for line in linesToAdd:
    trainingFile.write(line)
