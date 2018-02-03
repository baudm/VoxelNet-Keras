# VoxelNet-Keras
VoxelNet implementation in Keras

*NOTE*: This is still a work in progress. The full training pipeline is not yet done, but the model is practically fully implemented (see `model.png`) aside from some missing activations near the top Conv2D/Conv2DTranspose layers. Also, the sampling and grouping can be done as a preprocessing step outside the network, hence the difference in input shape vs what was described in the paper.
