# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""
# Jones matrix ----- DOI: 10.1103/PhysRevE.74.056607

import numpy as np

import GPdataset
import random
         
# one of the anisotropy vectors
V = [ [1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1] ]



def save_Object(path, lenDataset,noiseValue):
    dataset = []
    for i in range(lenDataset):
        polObject = GPdataset.GPdataset()
        classVector = random.choice(V)
        polObject.change_properties(classVector = classVector)
        polObject.calculation_Corr_Functions()
        parameters = np.array([],np.single)
        if noiseValue != 0.: polObject.noise_Simulation(noiseValue = noiseValue) # create a noise
        for par in polObject.setOfParametrs:
            parameters = np.append(parameters,par)
        dataset.append(([polObject.setOfCorrFunc,\
                           polObject.classVector, parameters]))
    np.save(path,dataset)
                 
    
    
    
    
if __name__ == "__main__":
    NumberOfObjects = 100
    Noise = 0. # value of noise from 0. to 1.
    NameOfFile = "testDataset"
    save_Object(NameOfFile,NumberOfObjects,Noise)
    
