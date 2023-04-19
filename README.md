# GPdatasetCreator
GPNN is a neural network for solving the ghost polarimetry problem in the formalism of Jones matrices.
This repository has programs to create the datasets required for GPNN training.


GPdataset.py - dataset class for GPNN training

GP_Generator.py - dataset generation software

  Parameters:
  
  NameOfFile      - path to file;
  
  NumberOfObjects - number of objects in dataset;
  
  Noise           - value of noise.




Step by step:

1. Download this rep.
2. Download Trained Weights for GPNN - https://drive.google.com/file/d/1ux1vYM6inBCmh2FVPyd0lG4Gf5smoTVt/view?usp=share_link .
3. Open "name" script and specify the path to the downloaded weights.
4. Specify the measured values of the correlation functions
5. Run name 


This work is supported by the Russian Science Foundation under grant No.21-12-00155.
