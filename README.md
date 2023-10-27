# GPdatasetCreator
GPNN is a neural network for solving the ghost polarimetry problem in the formalism of Jones matrices.
This repository has programs to create the datasets required for GPNN training.


GPdataset.py - dataset class for GPNN training

GP_Generator.py - dataset generation software

  Parameters:
  
  NameOfFile      - path to file;
  
  NumberOfObjects - number of objects in dataset;
  
  Noise           - value of noise.




Step by step to sart:

1. Open "GPNN_run.ipynb"
2. Download Your data or data from GPdataset.
3. Run "GPNN_run.ipynb"

The trained weights for GPNN are located in the repository in the folder "GPNN/Example/TrainedGPNN_openDataset"
The data is located in the repository in the folder: "GPNN/Example/OpenDataset_20k_1PercNoise.npy"

The program "GPNN_run.ipynb" visualizes the operation of a neural network. The program performs pixel-by-pixel processing of two-dimensional images. At the end of the program there is a data visualization. It can be seen that, with the exception of a few pixels, the program correctly predicts the class of polarization properties. This program can also be used to process other data.

This work is supported by the Russian Science Foundation under grant No.21-12-00155.
