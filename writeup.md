## Project: Follow Me

#### Steps to complete the project:

- Clone the project repo [here](https://github.com/udacity/RoboND-DeepLearning-Project.git)
- Fill out the TODO's in the project code in the `code/model_training.ipynb` file.
- Optimize your network and hyper-parameters.
- Train your network and achieve an accuracy of 40% (0.4) using the Intersection over Union IoU metric.
- Make a brief writeup report summarizing why you made the choices you did in building the network.

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points

---
### Writeup

[//]: # (Image References)
[image1]: ./docs/misc/model.png

#### Model Architecture

In this project, we want to locate where the target is in the scene so that the quadcopter can follow the target. We use Fully Convolutional Network (FCN) to achieve this goal.

A typical CNN is a great architecture to classify a picture whether it contains the target. However, since typical CNNs do not preserve the spacial information, we are not able to locate where the target is using this architecture. FCN, on the other hand, preserves spacial information throughout the entire network. This enables us to extract the target-pixels from the scene.

An FCN is usually comprised of two parts; encoder and decoder. The encoder is a series of convolutional layers like VGG and ResNet. Its goal is to extract features from the image. The decoder is also a series of convolutional layers. However its goal is to up-scale the output of the encoder to match its size with the input image. Thus, the overall output results in a pixel-by-pixel segmentation of the original image.

In addition to the up-sampling technic with the encoder/decoder structure, FCNs apply other two technics to make its segmentation  more precise; 1x1 convolutional layers and skip connection.

1x1 convolutional layers enable the network to multiply the sums of convolution in the encoder layer while preserving their spacial information. Skip connection works by connecting the output of one layer of the encoder to another layer of the decoder. This allows the network to use information from multiple resolutions.

In this project, the network is comprised of the following layers:

- 3 encoder layer layers (channels=32,64,128)

- 1x1 convolution layer (channels=256)

- 3 decoder layers (channels=128,64,32)

There are also input and output layers with 3 input channels (RGB) and 3 output channels (hero, human, background), respectively.

Although adding more layers may improve the segmentation result,  it increases the computational time on both training and prediction. Setting 3 layers for each encode/decode layer seems to be reasonable.

Blow is an plotted image of the model:

![model plot][image1]

#### Parameters

### Implementation

### Model

#### 1. The model is submitted in the correct format.

#### 2. The neural network must achieve a minimum level of accuracy for the network implemented.
