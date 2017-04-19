
# **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[LeNet]: ./examples/lenet.png "LeNet 5"
[bars]: ./examples/bars.png "traffic signs data sets"
[table]: ./examples/table.png "Categories of traffic signs"
[analysis]: ./examples/analysis.png "# of images per category"

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ASJ3/CarND_Term1_P2/blob/master/LeNet-Lab-Traffic-Signs-Normalized-Copy3.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python to calculate summary statistics of the traffic
signs data set:

* The size of training: 34799
* The size of the validation set: 4410
* The size of test set: 12630
* The shape of a traffic sign image: (32, 32, 3) which stands for width X height X depth (depth is 3 because of the 3 color channels)
* The number of unique classes/labels in the data set: 43

Here are descriptions of each unique traffic sign label in the data set: 
![alt text][table]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is broken down among the 3 sets

![alt text][bars]

Furthermore, an exploration of the number of images in the training set and testing set shows that not all traffic signs are represented equally. Some signs have far more examples that others. However, the proportion of each sign in the training and testing sets is about the same.

![alt text][analysis]


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[LeNet]: ./examples/lenet.png "LeNet 5"
[bars]: ./examples/bars.png "traffic signs data sets"
[table]: ./examples/table.png "Categories of traffic signs"
[inv]: ./examples/inverted_colors.png
[real]: ./examples/real_colors.png


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to normalize the images in all the sets (i.e. training, validation and testing). This was done by taking each pixel RGB value, subtracting 128 from it, then dividing it by 128. Each new pixel value, instead of being between 0 and 255, was thus between -1 and 1. I had read that normalizing the values would ensure that the model wouldn't have to try as hard to find a local minimum.

To ensure that my normalization kept the same proportions as the original I decided to use pyplot to show me back the images after the data was "de-normalized". This was done by simply reversing the operations on each pixel value: multiply by 128 and add 128. However when I tried to visualize a de-normalized image I got the following:


![alt text][inv]

Instead of this:

![alt text][real]

Although I thought this change in color rendering might be due to how matplotlib was rendering the image, and didn't have anything to do with my normalization process, I decided to investigate further. I found out the issue lies indeed with how matplotlib was interpreting the integers in the de-normalized data, as it deals differently with integers of type uint8 and int64 for example. A more detailed explanation on this can be found on the [answer I wrote on Stack Overflow](http://stackoverflow.com/questions/36137980/matplotlib-imshow-inverting-colors-of-2d-ifft-array/43269265#43269265)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is based mostly on LetNet 5 from Yann Lecun, with a few hyperparameters modifications, and consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 5x5 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 5x5 stride, same padding, outputs 10x10x16   	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
| Fully connected		| input: flatten 5x5x16 = 400, output: 120      |
| RELU					|												|
| Fully connected		| input: 120, output: 84                        |
| RELU					|												|
| Fully connected		| input: 84, output: 43                         |
| Softmax				| Output probabilities for 43 sign categories 	|


# LeNet Lab
![examples](./examples/lenet.png)
Source: Yan LeCun

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[LeNet]: ./examples/lenet.png "LeNet 5"
[bars]: ./examples/bars.png "traffic signs data sets"
[table]: ./examples/table.png "Categories of traffic signs"
[pic1]: ./examples/0_speed_limit_20kmh.jpg "20 Kmh speed limit"
[pic2]: ./examples/3_speed_limit_60kmh.jpg "60 Kmh speed limit"
[pic3]: ./examples/9_no_passing.jpg "No passing"
[pic4]: ./examples/14_stop.jpg "Stop"
[pic5]: ./examples/23_slippery_road.jpg "Slippery road“
[pic6]: ./examples/21_double_curve.jpg "Double curve"


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I broke the training set into 272 mini-batches using a batch size of 128 (because there were 34799 signs in the training set, the last mini-batch size was only 111).

The number of epochs in the final model was 30, as increasing that number tended to yield better accuracy, although I changed the learning rate according to the number of period (see next section for details).

I used two learning rates:
* an initial rate of 0.001
* a smaller rate of 0.0001

I initially thought about using Stochastic Gradient Descent for optimization, but my model is based on LeNet 5, which uses Adam optimizer. After doing research I learned Adam optimizer tends to converge faster than SGD because it uses momentum, so this is the optimizer I decided on using.

My model also uses two convolutional layers, the first one with a 5x5x3 shape and the second with a 5x5x6 shape. I left the stride of these filters unchanged from the LeNet model, with a stride of 1.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000 (see *Training Accuracy* section in iPython export)
* validation set accuracy of 0.942 (see *Training / Validating the Model* section in iPython export)
* test set accuracy of 0.931 (see *Test the Model* section in iPython export)

My model is greatly inspired by LeNet 5, as this model was created to categorize pictures of digits and so its architecture is also well suited to recognize other pictures. Also, the size of the image used in LeNet (32x32x1) is pretty similar to the traffic signs images (32x32x3).

In the end I had to change to 3 the depth of the filter in the first convolutional network, so that it would process the RGB pictures of traffic signs. I also had to change the number of categories in the output layer to 43 (the number of signs types in our data set).

After making those changes I ran the model without changing any of the hyperparameters (e.g. initial number of epochs were 10), to see what accuracy I could get. Initial results were pretty good (in the 80-85% range) although not enough to reach the desired 93% accuracy for the test set.

To increase accuracy I first decided to normalize each image. Normalization helped reach convergence faster and when running the model again, accuracy increased by about 4% for the validation model.

The other hyperparameters I tried to modify were:
* batch size (I tried batch sizes of 96, 128 and 256)
* number of epochs (between 10 and 30)
* learning rates (between 0.001 and 0.0001)

Changing the batch sise didn't prove conclusive. In fact, increasing the batch size to 256 significantly decreased the accuracy of the model with the validation set! After a few trials, I decided to leave batch size at 128, identical to the original LeNet 5 model.

Increasing the number of epochs definitely helped increase the accuracy, but usually after the 20th period I noticed some fluctuation of the accuracy on the validation set. Instead of seeing the accuracy steadily increase, it would fluctuate by 1 or 2% percentage points around the 92% mark. I thought this might have to do with the learning rate being too high and creating divergence resulting in the model always "overshooting" the local minimum.

This is the reason why I decided on creating two learning rates: an initial one at 0.001 which I would use to train the model at the beginning; and a smaller rate of 0.0001 used to help the model converge towards a local minimum at the end of the training. 

After trying different ratios of big v. small learning rate with regards to the number of epochs, I settled on using the big learning rate for the first 26 epochs, then using the small learning rate for the last 4 periods, for a total of 30 epochs.

My final model reached 94.2% accuracy with the validation set, and 93.1% accuracy with the test set.

 

### Testing the Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:
* Speed limit of 20 Kmh:

![alt text][pic1] 

* Speed limit of 60 Kmh:

![alt text][pic2] 

* No passing:

![alt text][pic3] 

* Stop:

![alt text][pic4]
 
* Slippery road:

![alt text][pic5]

The original images I found on the internet were of much higher resulotion and did not have the same width and height. So I resized each picture to 32x32 pixels. I also reviewed examples from the original data set to look at the average proportion of sign v. background on the image used. My goal was to make these 5 new pictures as close as possible to the images used to train the model, in order to increase the model prediction.

The first two signs are almost identical, so I wanted to see if the model could efficiently differentiate them. The third sign is also of the same shape as the first two signs. 

The stop sign was probably the one I thought would be the easiest to recognize (because its distinct shape is only used for one sign).

The last sign might have been a little trickier to recognize as the sign was slightly crooked. Again I wanted to see how the model fared in this situation.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit 20 kmh 	| Speed limit 20 kmh    						| 
| Speed limit 60 kmh 	| Speed limit 60 kmh 							|
| No passing			| No passing									|
| Stop sign	      		| Stop sign					 		     		|
| Slippery Road			| double curve      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares less favorably than the accuracy on the test set of 93%, although my data set is very small here.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the second-to-last cell of the Ipython notebook.

For the first image, the model is extremely sure that this is a 20 kmh speed limit sign (probability of 0.9995), and the image does contain that sign. It is also interesting to note that most of the other top probabilities are signs of different speed limits. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9995         		| Speed limit 20 kmh   							| 
| .0004     			| Speed limit 30 kmh 							|
| ~.0					| Speed limit 70 kmh							|
| ~.0	      			| Speed limit 80 kmh					 		|
| ~.0				    | Bycicles crossing      						|

For the second image, the model is extremely certain that this is a 60 kmh speed limit sign (probability of 0.9994), and the image does have that sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9994         		| Speed limit 60 kmh  							| 
| .0006     			| Speed limit 50 kmh							|
| ~.0					| Speed limit 30 kmh							|
| ~.0	      			| Wild animals crossing					 		|
| ~.0				    | Speed limit 20 kmh     						|


For the third image, the model is also extremely certain that this is a no passing sign (probability of 0.99998), and the image does contain that sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99998         		| No passing        							| 
| .00002     			| End of no passing	    						|
| ~.0					| Dangerous curve to the right					|
| ~.0	      			| No passing for vehicles over 3.5 tons			|
| ~.0				    | Children crossing     						|

For the fourth image, a stop sign, the model's top probability was so high that python rounded it to 1.0, and the image does contain that sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0           		| Stop sign         							| 
| ~.0       			| Speed limit 60 kmh   	   						|
| ~.0					| No vehicles               					|
| ~.0	      			| Speed limit 50 kmh		                    |
| ~.0				    | Speed limit 80 kmh     						|


The last image was a slippery road sign and the model didn't recognize it as such but predicted instead a double curve (probability of 0.91). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .910           		| Double curve       							| 
| .046       			| Slippery road     	   						|
| .034					| Wild animals crossing              			|
| .009	      			| Dangerous curve to the left		            |
| ~.0				    | Right of way at the next interscction			|

So for the last image the model thought the sign was a double curve:

![alt text][pic6] 

Instead of a slippery road:

![alt text][pic5] 


## Conclusion:
LeNet 5 performed very well once the pictures were normalized. This goes on to show the adaptability of a convolutional neural network. Although LeNet was designed to recognize a digit (out of 10 possible categories) it also worked to identify a traffic sign among 43 categories. Granted, both image sets (in LeNet and traffic signs) had images of the same width and height, so with larger traffic-sign pictures the architecture of the LeNet network would probably have to be modified, most notably by increasing the size of the filters for the two convolutional layers. It would be interesting then to see if images of higher resolution bring better prediction than low-resolution, blurry images as the ones used in the traffic-sign data set.

