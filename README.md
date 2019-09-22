# Multimodal-Classification
Multimodal Classification of image and text using Deep Learning.
Our model aims to classifiy the image and text data into 6 damage classes usin CNN: 1)damaged infrastructure 2)damged_nature 3)fires 4)flood 5)human_ damage 6)non_damage

Then we aim to build a multimodal frame work which idenitfies damage related information containing both text and images.

The framework combines multiple pretrained unimodal CNN n
that extract features from text and images inpendently and  train in it a diffusion model.

 Feature fusion (FF) combines the internal representations generated from the various layers of the CNNs to train a multimodal classifier.
 Decision fusion (DF) combines the predictions of the unimodal classifiers to obtain better predictions by either using decision rules or training a classifier


Dataset: new_multimodal
**files should be run in the respective order.

text_revised file trains on textual data in the given 6 classes and evaluate the loss and accuracy on validation set.

Image mode file trains inception module CNN model  on the image data and evaluate loss and accurarcy.

ff model concatenate the the layer before softmax of both image and text model and then train it on 4 layer CNN using multi input(text,image) keras functional api.

df model adds six-dimensional softmax outputs of image and text model and then apply softmax to predict maximum decision output , where the decision is based on the class corresponding to the index with the maximum value.

NOTE:image model and text revised must be run before running df model.
