# transfer learning for image data

In transfer learning, the knowledge(weight) of an pre-trained model is applied to other problems. Other problems may be similar or different or could be exact to task of pre-trained model. We want to improve generalization.

If the second task is exactly same to task of pre-trained model we can use TL_as_classifier method.
If the second task is similar to task of pre-trained model we can use TL_as_feature_extractor method to extract features or to be exact general features from earlier and middle layers to later useses for new task.
If the second task is different from the task of pre-trained model we can use TL_as_wieght_initializtor method to combine pre-trained model and our model for new task. Both our model and pre-trained model weights will be changed durign learning. We use this method for weight initialization.
