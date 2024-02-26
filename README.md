# Dog vs Cats ML Model

This repository contains a machine learning model that classifies images as either dogs or cats. The model is trained on a dataset containing images of dogs and cats.

## Dataset
The dataset consists of thousands of images of dogs and cats. It is divided into two classes: dogs and cats. Each class contains an equal number of images for balanced training.

## Model Architecture
The machine learning model is built using a convolutional neural network (CNN). The architecture of the CNN includes several convolutional layers followed by max-pooling layers for feature extraction. The final layers consist of fully connected layers with dropout regularization to prevent overfitting. The model is trained using the Adam optimizer with categorical cross-entropy loss.

## Training
The model is trained on a GPU for faster processing. The training process involves feeding batches of images through the network and adjusting the model weights based on the error calculated using the categorical cross-entropy loss function. Training is performed over multiple epochs until the model converges and achieves satisfactory accuracy on the validation set.

## Evaluation
The performance of the model is evaluated using a separate test set that was not seen by the model during training. Evaluation metrics such as accuracy, precision, recall, and F1-score are calculated to assess the model's performance in classifying dogs and cats correctly.

