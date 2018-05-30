# **Traffic Sign Recognition Writeup**  
Shuo Feng
---

##Goals of this project:

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: writeup_images/sampleImage.png "Sample Images with Class Labels"
[image2]: writeup_images/hist.png "Histogram of the 43-Class Distribution"
[image3]: writeup_images/grayscale.png "Raw RGB Image and the Normalzed Gray Scale Image"
[image4]: writeup_images/fiveInternetImages.png "Raw RGB Image and the Normalzed Gray Scale Image"
[image5]: writeup_images/confidence.png "confidence of 5 images"
[image6]: writeup_images/layer1_weight.png "First Layer Weight"


[img1]: images/img1.png "Image 1 From Internet"
[img2]: images/img2.png "Image 2 From Internet"
[img3]: images/img3.png "Image 3 From Internet"
[img4]: images/img4.png "Image 4 From Internet"
[img5]: images/img5.png "Image 5 From Internet"


---
## Project Summary

The source code of my project can be found in this [github repo](https://github.com/StevenShuoFeng/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb).

## Data Set Summary & Exploration

#### 1. Summary of the dataset. 
The data is downloaded from the courseware, and the data files contains pickled file for training, validation and testing file, one for each and a csv file of the class names.

The size of the training, validation and testing data are:
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630

And each sample of all the above data sets is a 32x32x3 RGB image representing traffic signs of 43 categories. 
A few example of the sample images are given below:
![alt text][image1]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training data set. The following char is the normalized historgram of data count of each of the 43 categories. For example, the class label 2 and 3 make up about 5.5% of the training data, while the class label 20 is less than 1%. So the data is seriously unbalanced.

![alt text][image2]

## Design and Test a Model Architecture

#### 1. Data Processing

The preprocessing of the images is simple, each 3-channel RGB image is mapped to a 1-channel gray scale by adding 3 channels with weight of 0.299, 0.587, 0.114 correspondingly. Then the image is subtracted by 128 and normalized by 128. After this, most images have mean value close to 0, and the value of all pixels are within range of [-1, 1]. A sample  of the RGB image and the corresponding gray scale image is given below:

![alt text][image3]

No data augmentation is made although I wish I have time to do it, especially for those classes with much less examples.


#### 2. Final Model Architecture

The model is adapted from the paper [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) as suggested by the notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|-----------------------|-----------------------------------------------|
| Input         		| 32x32x1 Gray Scale image   					|  
|-----------------------|-----------------------------------------------| 
| *Layer 1*				|												|
| Input         		| 32x32x1 										| 
| Convolution 1 5x5 	| 1x1 stride, valid padding, output 28x28x6  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride									|
| Output 		      	| Layer 1 Output 14x14x6						|
|-----------------------|-----------------------------------------------| 
| *Layer 2*				|												|
| Input         		| 14x14x6   									| 
| Convolution 2 5x5 	| 1x1 stride, valid padding, output 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride									|
| Output		      	| Layer 2 Output 5x5x16							|
|-----------------------|-----------------------------------------------| 
| *Layer 3*				|												|
| Input         		| 5x5x16	  									| 
| Convolution 3 5x5 	| 1x1 stride, valid padding, output 1x1x400  	|
| RELU					| 												|
| Output				| Layer 3 Output 1x1x400 						|
|-----------------------|-----------------------------------------------| 
| *Layer 4* 			| Flatten and combine the output of Layer 2&3	|
| Flatten Layer 2 Output| 5x5x16 >>> 400								| 
| Flatten Layer 3 Output| 1x1x400 >>> 400								| 
| Output		 		| Layer 4 Output 800							| 
|-----------------------|-----------------------------------------------| 
| Fully connected		| Input 800	Output 43-Class						| 
| dropout 				| keep_probability = 0.75						| 

It's defined as 'LeNet_YannLe' funciton in the code. The size of each layer is printed out in the code as below:
layer 1 shape: (?, 14, 14, 6)
layer 2 shape: (?, 5, 5, 16)
layer 3 shape: (?, 1, 1, 400)
layer2flat shape: (?, 400)
layer3flat shape: (?, 400)
FC input shape: (?, 800)

#### 3. Model Training

The model is trained using 'AdamOptimizer' with 30 epochs and batch size of 128. Learning rate is set to 0.001. 

#### 4. Tuning process:

The LeNet was used first as it's directly available from the course material. However, the accuracy is bounded to about 93%

Then, refering to the paper with multi-scale feature, another architecture is built. With the same training data, a 2% increase in validation accuracy is easily gained.
After that, the major change is change epoch from about 10 to 30~50. 
The training was originally done on google cloud CPU which is extremely slow. After setup the AWS gpu instance, the process is much faster. Learning rate and dropout ratio, the validation accuracy reaches 96.3%

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 96.3%
* test set accuracy of 93.9%


### Test a Model on New Images

#### 1. Five Images from Internet

Here are five German traffic signs that I found on the web:

![alt text][img1]
![alt text][img2] 
![alt text][img3] 
![alt text][img4] 
![alt text][img5]

The second 'Stop Ahead' sign and the 'Deers Showing Up' sign are hard to classify as they're not shown up much in the training categories. 
These images are downsampled to 32x32x3 and pre-processed in the same way as the other training data sets as below:
![alt text][image4]

#### 2. Discussion of Prediction

Here are the results of the prediction:

| Image True Class      | 
|-----------------------|
| Stop Sign      		|
| Stop Ahead 			|
| Turn right 			|
| Deer Around      		|
| Children Crossing		|

|     Prediction	        							| 
|-----------------------------------------------		| 
|image # 1 prediction:  1 Speed limit (30km/h)			|
|image # 2 prediction:  1 Speed limit (30km/h)			|
|image # 3 prediction:  33 Turn right Ahead 			|
|image # 4 prediction:  19 Dangerous curve to the left	|
|image # 5 prediction:  11 Right-of-way at the next intersection|


Only the 'Turn Right' sign  is correctly classified and the accuracy is 20%. Compare to the 94% accuracy on testing data set, this is quite low. However, considering that the 2nd and 4th image may not be seen in the training data set, this is kind of expected.
But I doubt that the image I cropped are taking bigger potion of the field of view than those in the training data set.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, it's 60% classified as label 1. The probability of the network result for each image is show below:
![alt text][image5]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
The first layer (14x14x6) is visualized here and they're trying to find the edges of the signs.
![alt text][image6]


