from keras.layers import Dense
from keras.models import Model
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

def TL_as_classifier(img_dir):
    image = load_img(img_dir, target_size=(224, 224))
    image = img_to_array(image)
    
    #vgg16: expected shape=(None, 224, 224, 3)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    #vgg16 needs image to be preprocessed
    image = preprocess_input(image)
    
    target = decode_predictions(VGG16().predict(image))[0][0]
    
    return target[1], target[2]
    

def TL_as_feature_extractor(img_dir):    
    image = load_img(img_dir, target_size=(224, 224))
    image = img_to_array(image)
    
    #vgg16: expected shape=(None, 224, 224, 3)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    #vgg16 needs image to be preprocessed
    image = preprocess_input(image)
    
    # This is model without output or classifier layer (include_top=False)
    vgg_model = VGG16(include_top=False)
        
    extracted_features = vgg_model.predict(image)
    
    return extracted_features
    
def TL_as_wieght_initializtor(no_dnese_out, no_class):
    
    # This is model without output or classifier layer (include_top=False)
    vgg_model = VGG16(include_top=False, input_shape=(300, 300, 3))
    
    # input of flatten is vgg_model.layers[-1].output
    flatten = Flatten()(vgg_model.layers[-1].output)
    
    #input of fully_connected is flatten
    fully_connected = Dense(no_dnese_out, activation='relu')(flatten)
    
    # input of classifier is fully_connected
    classifier = Dense(no_class, activation='softmax')(fully_connected)
    
    model = Model(inputs=vgg_model.inputs, outputs= classifier)
    
    return model
