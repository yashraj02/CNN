# CNN-Animals_Classification

## Transfer Learning
What is Transfer Learning?

Due to the fact that training the model needs to take a long time even with high-end GPUs, reuse the proved model is a better idea. Avoiding to spend much time on training and verification, transfer learning is the necessary method what we need.

Transfer learning is borrowing architecture from its pre-trained parameters from others model (which may or may not heve been trained on similar data). Which means we can train our own data and can easily to reach the accuracy of the target. For CNN as an example in the story:

Load a pre-trained CNN model & detach the FC layers from it.
Example : 
from keras.applications import VGG16
model = VGG16(weights="imagenet", include_top=False)

Freeze the weights or parameters of the model.
Replace the layers of the original model with customer layers.
Train the layers on training data

Once trained you can store the model using model.save("my_model.h5") into HDF5.

Models Trained based on Transfer Learning can be expected to outperform in most secenarios. For code kindly look up to Transfer-Learning directory in CNN(current repo) repository.

Transfer learning helps when you are low on configs to run high task oriendted algorithms from scrath.

## HDF5
HDF5 is binary data format created by the HDF5 group [12] to store gigantic numerical datasets on disk (far too large to store in memory) while facilitating easy access and computation on the rows of the datasets. Data in HDF5 is stored hierarchically, similar to how a file system stores data. Data is first defined in groups, where a group is a container-like structure which can hold datasets and other groups. Once a group has been defined, a dataset can be created within the group. A dataset can bethought of as a multi-dimensional array (i.e., a NumPy array) of a homogeneous data type (integer, float, unicode, etc.).

We can store huge amounts of data in our HDF5 dataset and manipulate the data in a NumPy-like fashion. 
These slices and row accesses are lighting quick. When using HDF5 with h5py, you can think of your data as a gigantic NumPy array
that is too large to fit into main memory but can still be accessed and manipulated just the same.

Uses:
1. Facilitate a method for us to apply transfer learning by taking our extracted features from
VGG16 and writing them to an HDF5 dataset in an efficient manner.
2. Allow us to generate HDF5 datasets from raw images to facilitate faster training.
