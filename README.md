# Image-Classification

Problem Statement:                                                                                                                        
Given a set of labelled images, train a classification model to classify the labels for images.The dataset consists of the URLs for the training and test images.

Developed 3 models for the problem statement:
1. A CNN model in tensorflow from scratch
2. A pre-trained keras model, with just re-training the output dense softmax layer
3. A pre-trained keras model, with keeping the initial few lower layers freezed (weights won't change during training), and for the remaining architecture, training again all the remaining layers.

The repo also contains the SQL query used to extract the table, having the labels, URLs, and whether train or test image, from Google Cloud BigQuery, and also a brief summary report of the entire project, which documented the steps taken, and also the metrics values achieved.
