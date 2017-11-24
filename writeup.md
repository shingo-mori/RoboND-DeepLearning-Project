## Project: Follow Me

#### Steps to complete the project:

- Clone the project repo [here](https://github.com/udacity/RoboND-DeepLearning-Project.git)
- Fill out the TODO's in the project code in the `code/model_training.ipynb` file.
- Optimize your network and hyper-parameters.
- Train your network and achieve an accuracy of 40% (0.4) using the Intersection over Union IoU metric.
- Make a brief writeup report summarizing why you made the choices you did in building the network.

#### Project Specification
 Rubric points for this project are explained [here](https://review.udacity.com/#!/rubrics/1155/view).

---
### Writeup

- TODO: write limitations and improvements
- TODO: write introduction

[//]: # (Image References)
[image_model]: ./docs/misc/model.png
[image_following_images]: ./docs/misc/following_images.png
[image_patrol_non_targ]: ./docs/misc/patrol_non_targ.png
[image_patrol_with_targ]: ./docs/misc/patrol_with_targ.png

#### Model Architecture

In this project, we want to locate where the target is in the scene so that the quadcopter can follow the target. We use Fully Convolutional Network (FCN) to achieve this goal.

A typical CNN is a great architecture to classify a picture whether it contains the target. However, since typical CNNs do not preserve the spacial information, we are not able to locate where the target is using this architecture. FCN, on the other hand, preserves spacial information throughout the entire network. This enables us to extract the target-pixels from the scene.

An FCN is usually comprised of two parts; encoder and decoder. The encoder is a series of convolutional layers like VGG and ResNet. Its goal is to extract features from the image. The decoder is also a series of convolutional layers. However its goal is to up-scale the output of the encoder to match its size with the input image. Thus, the overall output results in a pixel-by-pixel segmentation of the original image.

In addition to the up-sampling technic with the encoder/decoder structure, FCNs apply other two technics to make its segmentation  more precise; 1x1 convolutional layers and skip connection.

1x1 convolutional layers enable the network to multiply the sums of convolution in the encoder layer while preserving their spacial information. Skip connection works by connecting the output of one layer of the encoder to another layer of the decoder. This allows the network to use information from multiple resolutions.

In this project, the network is comprised of the following layers:

- input layer (channels=3 (RGB))

- 3 encoder layer layers (channels=32,64,128)

- 1x1 convolution layer (channels=256)

- 3 decoder layers (channels=128,64,32)

- output layer (channels=3 (Hero,Human,Background))

Although adding more layers may improve the segmentation result,  it increases the computational time on both training and prediction. Setting 3 layers for each encode/decode layer seems to be reasonable.

Blow is an plotted image of the model:

![model plot][image_model]

#### Training
The model is trained using default datasets provided by Udacity.

Hyper-parameters used for training the model are shown below:
```py
learning_rate = 0.005
batch_size = 32
num_epochs = 20
steps_per_epoch = 150
validation_steps = 50
workers = 4
```

Setting learning_rate to 0.005 seemed to be reasonable for both final score and the learning speed.

Number of epochs is set to 20, since the final score does not increased much after 20 epochs.

Other hyper-parameters (batch_size, steps_per_epoch, validation_steps, workers) are determined based on the hardware used in this project. (GeForce GTX 1060 3 GB)

The model is trained using Adam optimizer. A multi-class [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy) is used to calculate the training and validation error.

#### Result
The trained model is evaluated using the following three different types of test-images:

- following_images
  - Images to test how well the network can identify the target while following them.
- patrol_with_targ
  - Images to test how well the network can detect the hero from a distance.
- patrol_non_targ
  - Images to test how often the network makes a mistake and identifies the wrong person as the target.

Example images of input image, ground truth, and predicted image by the trained model are shown below:

- following_images

![image_following_images][image_following_images]

- patrol_with_targ

![image_patrol_with_targ][image_patrol_with_targ]

- patrol_non_targ

![image_patrol_non_targ][image_patrol_non_targ]

The performance of the model is measured by true-positive ratio and  Intersection over Union (IoU) metric. The final score of the model reached 0.411516123844, which is above the required score of 0.4.

#### Limitations and Improvements

#### Files
##### model
The trained weights of the model are saved in the HDF5 format. They are located in the following directory:

```
./data/weights/
  config_model_weights1
  model_weights1
```

##### notebook
The model is implemented in the below notebook.  There is also a html version  of this notebook:

```
./code/
  model_training.ipynb
  model_training.html
```
