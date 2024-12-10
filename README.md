# Traffic-Sign-Detection
In the fast-growing field of autonomous driving, the detection of road signs should be both fast and precise. Traditional methods for the detection of road signs are usually supported by complex machine learning models, such as deep neural networks, which can be quite computationally expensive. For instance, HydraNet is a neural network that accomplishes multiple tasks simultaneously, including the detection of traffic lights, lane markings, and other vehicles, stitching multiple camera angles, and processing image data from eight different cameras in Tesla's road sign detection system. Some techniques of sketch recognition, such as heuristic-based classification and template matching, can reduce the computational burden while enhancing the speed of processing. These methods use heuristics to quickly identify road signs by comparing real-time images with a library of templates, without going into deep computation.
Our work leverages heuristic identification techniques, including stroke feature and corner detection combined with the template matching of sketch recognition, to reduce the computational load but maintain high accuracy. To evaluate our method, we compared the speed and accuracy of our model with traditional deep learning models for road sign detection. Both models were tested using the same set of real-time images to measure classification speed and accuracy. For processing time, our approach gave us 0.050 seconds against 0.21 seconds achieved by the basic deep learning model. In terms of performance, our approach achieved an F1 score of 0.69 compared to a score of 0.732 attained by the deep learning model. This F1 score is slightly lower, but such drastic cutback in processing time has proved to be a compelling bargain.
Our research would wish to contribute to the study of autonomous driving by offering a different approach to road sign detection that reduces computational time. It also lays a foundation for further studies in developing superior hybrid models that combine template matching and deep learning techniques.

# Prerequisites
1. Python
2. OpenCV
3. Numpy
4. Sklearn
5. Matplotlib
   
# Steps to run main.py
1. First download the main.py file.
2. Then download the two zip files in the repository: traffic_signs_dataset and final_templates_project
3. Unzip these files and make sure that the folder path is known.
4. Then we open the main.py file and add the folder path to the unzipped files shown by the comments.
5. Then we run the code. Make sure you have python and the other necessary imports for running the file, otherwise please install it.
6. We will get a classification report and a sample for which time was calculated for classification along with the confusion matrix for the same.
7. We have a report where the result analysis is discussed in details

There are other files in the repository: the CNN_sample_model.ipynb file, which was a model CNN code used for comparision to our model, and a web_app.ipynb file which was an temporary webapp using just heuristics for analysis of roadsigns.
