# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""
# Jones matrix ----- DOI: 10.1103/PhysRevE.74.056607

import numpy as np
import random 
#Jones matrix of Linear amplitude anisotropy(LAA)
def matrixLAA(theta = 0, P = 1, randomMod = True):
    if randomMod: 
        P = random.uniform(0,1)
        theta = random.uniform(-np.pi/2,np.pi/2,)
    M11 = np.cos(theta)**2 + P*np.sin(theta)**2
    M12 = M21 = (1-P)*np.cos(theta)*np.sin(theta)
    M22 = np.sin(theta)**2 + P*np.cos(theta)**2
    return np.array([[M11,M12],[M21,M22]]), theta, P

#Jones matrix of Linear Phase Anisotropy (LPA).
def matrixLPA(alpha = 0, delta = 0, randomMod = True):
    if randomMod: 
        delta = random.uniform(0.,np.pi)
        alpha = random.uniform(-np.pi/2,np.pi/2,)
    M11 = np.cos(alpha)**2 + np.exp(-1j*delta)*np.sin(alpha)**2
    M12 = M21 = (1-np.exp(-1j*delta))*np.cos(alpha)*np.sin(alpha)
    M22 = np.sin(alpha)**2 + np.exp(-1j*delta)*np.cos(alpha)**2
    return np.array([[M11,M12],[M21,M22]]), alpha, delta

#Jones matrix of Circular Amplitude Anisotropy (CAA)
def matrixCAA(R = 0,randomMod = True):
    if randomMod: R = random.uniform(-1,1)
    M11 = M22 = 1.
    M12 = -1j*R
    M21 = 1j*R
    return np.array([[M11,M12],[M21,M22]]), R

#Jones matrix of Circular Phase Anisotropy (CPA)
def matrixCPA(phi = 0,randomMod = True):
    if randomMod: phi = random.uniform(0.,np.pi)
    M11 =  M22 = np.cos(phi)
    M12 = np.sin(phi)
    M21 = -np.sin(phi)
    return np.array([[M11,M12],[M21,M22]]), phi



class GPdataset():
    
    def __init__(self):
        self.jonesMatrix = np.array([[1.,0.],[0.,1.]],np.single)
        
        #All possible parameters of polarization sensitive objects.
        #Linear phase anisotropy(LPA)
        self.theta_LPA = np.nan
        self.value_LPA = np.nan
        #Linear amplitude anisotropy(LAA)
        self.theta_LAA = np.nan
        self.value_LAA = np.nan
        #Circular amplitude anisotropy(CAA)
        self.value_CAA = np.nan
        #Circular phase anisotropy(CPA)
        self.phi_CPA = np.nan
        
        self.transmission = 1. # isotropic coefficient of transmission
        self.classVector = np.array([0,0,0,0]) # this vector describe class of object 
        self.setOfCorrFunc = np.nan # set of value correlation functions 
        self.setOfParametrs = np.array([self.theta_LAA, self.value_LAA,\
                                        self.theta_LPA,self.value_LPA, self.value_CAA,\
                                            self.phi_CPA, self.transmission],np.single)
            # set, which contains  all  parametrs
            
    #This function randomly changes the properties of an object
    def change_properties(self, classVector = 0 ):
        

        if classVector == 0:
            self.classVector = np.random.randint(2, size = (1,4))[0]
        else:
            self.classVector = classVector    
            
        jonesMatrixLAA =    np.array([[1.,0.],[0.,1.]],np.single)
        jonesMatrixLPA =    np.array([[1.,0.],[0.,1.]],np.single)
        jonesMatrixCAA =    np.array([[1.,0.],[0.,1.]],np.single)
        jonesMatrixCPA =    np.array([[1.,0.],[0.,1.]],np.single)
            
        if self.classVector[0] == 1:
            jonesMatrixLAA, self.theta_LAA, self.value_LAA = matrixLAA()
        else:
            self.theta_LAA = self.value_LAA = np.nan
            
        if self.classVector[1] == 1:
            jonesMatrixLPA, self.theta_LPA, self.value_LPA = matrixLPA()
        else:
            self.theta_LPA = self.value_LPA = np.nan
            
        if self.classVector[2] == 1:
            jonesMatrixCAA, self.value_CAA = matrixCAA()
        else:
            self.value_CAA = np.nan
            
        if self.classVector[3] == 1:
            jonesMatrixCPA, self.phi_CPA = matrixCPA()
        else:
            self.phi_CPA = np.nan
            
        self.setOfParametrs = np.array([self.transmission, self.theta_LAA, self.value_LAA,\
                                        self.theta_LPA,self.value_LPA, self.value_CAA,\
                                            self.phi_CPA],np.single)
            
        self.jonesMatrix = self.transmission*jonesMatrixCPA.dot(jonesMatrixLPA).\
                                dot(jonesMatrixCAA).dot(jonesMatrixLAA)
        
    #This function calculates all normalized correlation functions        
    def calculation_Corr_Functions(self):
        g1 = abs(self.jonesMatrix[0,0])**2
        g2 = abs(self.jonesMatrix[1,0])**2
        g3 = abs(self.jonesMatrix[1,1])**2
        g4 = 0.5*abs(self.jonesMatrix[0,0]+self.jonesMatrix[1,0])**2
        g5 = 0.25*abs(self.jonesMatrix[0,0]+self.jonesMatrix[1,0]+1j*\
                      (self.jonesMatrix[1,1]+self.jonesMatrix[0,1]))

            
        self.setOfCorrFunc =  np.array([g1,g2,g3,g4,g5],np.single)
        
    #This function emulates noise (noiseValue < 1.)     
    def noise_Simulation(self, noiseValue = 0.):
        for i in range(len(self.setOfCorrFunc)):
            self.setOfCorrFunc[i] = self.setOfCorrFunc[i] + random.uniform(-noiseValue,noiseValue)
            
        
                 
        