import matplotlib.pyplot as plt

def show_most_simm(Xs, fig_row, fig_col, sz=10, show=True):
  
    plt.figure(figsize=(sz,sz))
    for i in range(0, ((fig_row * fig_col)+1)):
        try:
            plt.axis('off')
            plt.subplot(fig_row, fig_col, i+1)   
            plt.imshow(Xs[i]/255)
            
        except:
            pass
    if show:
        plt.show()

from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.models import Sequential, Model
from keras.optimizers import SGD,Adam, RMSprop

def createModelCNN(name, input_shape, number_of_classes, hiddens=[]):
    weights = 'imagenet'
    input_tensor = Input(shape=input_shape)
    if name == 'MobileNetV2':
        base_model = MobileNetV2(
            include_top=False,
            weights = weights,
            input_tensor=input_tensor,
           input_shape=input_shape,
            pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True  # trainable has to be false in order to freeze the layers
    
    model_concat = base_model.output
    for number_hidden in hiddens:
        model_concat = Dense(number_hidden)(model_concat)
        model_concat = Activation('relu')(model_concat)
        
    model_concat = Dense(number_of_classes)(model_concat)
    model_concat = Activation('softmax')(model_concat)
    model = Model(inputs=input_tensor,outputs=model_concat, name="ocr_"+name)
    return model

def createLossFunction(model, lr = 0.01, dr=0.01, m=0.9):
    learning_rate =lr
    decay_rate = dr
    momentum = m
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    #sgd = SGD(lr=0.000001, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
 #   model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
    #model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    try:
        model2 = multi_gpu_model(model, gpus=2)
        del model
        model = model2
    except:
        print("Error pallarel")
        pass
    return model

print("Import CNN Utility")