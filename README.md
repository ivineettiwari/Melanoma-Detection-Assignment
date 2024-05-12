# Melanoma Skin Cancer Detection

## Abstract
 Melanoma, constituting one of over 200 variants of cancer, presents a formidable challenge due to its propensity for aggressive progression if not detected early. Diagnosis typically commences with a triphasic approach: clinical screening, dermoscopic analysis, and histopathological examination. Melanoma, especially, necessitates swift and accurate identification for effective treatment.

 The initial stage involves a visual inspection of the skin lesion. Dermatologists employ high-speed cameras to capture dermatoscopic images, which serve as critical diagnostic aids. These images exhibit an inherent accuracy ranging from 65% to 80% in melanoma diagnosis, albeit without supplementary technical support.

 Enhancing diagnostic precision entails subsequent evaluation by cancer treatment specialists, complemented by the analysis of dermatoscopic images. This combined approach elevates the overall prediction rate to an accuracy bracket of 75% to 84%.

 In pursuit of optimizing melanoma diagnosis, there exists a project aimed at constructing an automated classification system. This system leverages image processing techniques to analyze and classify skin cancer using dermatoscopic images.

## Problem Statement

 In the context of skin cancer diagnosis, particularly melanoma, where early detection is crucial for effective treatment, the conventional process of obtaining a biopsy report is time-consuming, often taking up to a week or more from initial consultation with a dermatologist to receipt of the biopsy results. To address this delay and expedite diagnosis, a predictive model is proposed to narrow this timeframe to just a couple of days.

 This model employs a Convolutional Neural Network (CNN), a type of deep learning algorithm particularly adept at image classification tasks, to categorize nine distinct types of skin cancer from outlier lesions images. By leveraging the computational power and pattern recognition capabilities of CNNs, the aim is to streamline the diagnostic process and significantly reduce the time required for diagnosis.

 In contrast to the traditional approach, which relies on visual inspection, dermoscopic analysis, and histopathological examination, the proposed model offers a faster and potentially more accurate alternative. By automating the classification process based on image processing techniques, it has the potential to enhance diagnostic efficiency and positively impact the prognosis of millions of individuals affected by skin cancer, particularly melanoma, the most lethal form among over 200 variants of the disease.

## Motivation
 The primary objective of this initiative is to contribute to the reduction of mortality rates attributed to skin cancer. The project's core motivation lies in harnessing advanced image classification technology to enhance the well-being of individuals. With the rapid advancements in machine learning and deep learning, particularly in the field of computer vision, there exists significant potential for scalable solutions across various domains, including healthcare. By leveraging these cutting-edge technologies, the project endeavors to improve early detection and diagnosis of skin cancer, ultimately aiming to save lives and alleviate the burden of this disease on affected individuals and their families.

## Dataset
The International Skin Imaging Collaboration (ISIC) provided 2357 images of benign and malignant oncological disorders, which make up the dataset. Every image was arranged in accordance with the ISIC categorization, and each subset had an equal amount of photographs.

The following illnesses are included in the data set:

![datasetplot](https://raw.githubusercontent.com/ivineettiwari/Melanoma-Detection-Assignment/main/images/1.png)

Added extra examples across all classes so that none have extremely few samples using the Python module Augmentor (https://augmentor.readthedocs.io/en/master/) to address the problem of class imbalance.

### Sample image from Dataset

![sample image](https://raw.githubusercontent.com/ivineettiwari/Melanoma-Detection-Assignment/main/images/download.png)

## CNN Architecture Design
To use photos of skin lesions to categorize skin cancer. In order to improve performance and accuracy on the classification challenge, I developed a bespoke CNN model.

- Rescaling Layer: This layer is responsible for transforming input values from the [0, 255] range to the normalized [0, 1] range, facilitating consistent processing across the network.

- Convolutional Layer: Convolutional layers apply convolution operations to input data, merging information within local receptive fields to produce a single output value. This process reduces input dimensionality while consolidating information into individual pixels or features.

- Pooling Layer: These layers serve to downsample feature maps, reducing spatial dimensions and thereby decreasing the number of parameters and computational workload in subsequent layers. Pooling summarizes feature representations within defined regions of the input.

- Dropout Layer: Dropout layers randomly deactivate input units during training, reducing interdependencies between neurons and preventing overfitting by introducing regularization.

- Flatten Layer: This layer transforms multidimensional data into a one-dimensional array, facilitating the transition from convolutional layers to fully connected layers. It creates a continuous feature vector for input into subsequent layers.

- Dense Layer: Dense layers consist of interconnected neurons, with each neuron receiving input from all neurons in the preceding layer. They contribute to the network's capacity for complex pattern recognition and feature learning.

- Activation Function (ReLU): The rectified linear activation function (ReLU) allows direct output of positive inputs while zeroing negative inputs. Its piecewise linear nature mitigates the vanishing gradient problem, enhancing model learning efficiency and performance.

- Activation Function (Softmax): Softmax functions are typically employed in the output layer of neural networks for predicting multinomial probability distributions. They ensure output probabilities are constrained between 0 and 1, with the sum of all probabilities equating to one, facilitating interpretation and comparison of class likelihoods.

### Model Architecture
![Model Arch](https://raw.githubusercontent.com/ivineettiwari/Melanoma-Detection-Assignment/main/images/Screenshot.png)

### Model Evaluation
![ModelEvaluation](https://raw.githubusercontent.com/ivineettiwari/Melanoma-Detection-Assignment/main/images/download_2.png)