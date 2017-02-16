import numpy as np
import os,sys
from scipy import misc
import glob

def getImageByName(path):
    outputImage = misc.imread(path)
    return outputImage

def main():
    ##Get Training Data
    train_x_Dict = dict()
    train_y_Dict = dict()
    for label in ['ALB']:
        nameList = glob.glob('./data/train/%s/*.jpg' % label)
        for nameString in nameList:
            name = nameString.split('/')[-1]
            train_x_Dict.update({name: getImageByName(nameString)})
            train_y_Dict.update({name: label})
        print(label)

    return train_x_Dict, train_y_Dict

x, y = main()