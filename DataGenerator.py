from keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np
import gc
from skimage.transform import resize
from keras.utils import np_utils

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, Xs, ys, keras_image_generator=None , batch_size=32,
                 n_channels=3, n_classes=1, 
                 shuffle=True , image_size=128, list_IDs=None):
        'Initialization'
        if keras_image_generator is None:
            keras_image_generator =  ImageDataGenerator(rotation_range=0,
                               width_shift_range=0.0,
                               height_shift_range=0.0,
                               shear_range=0.00,
                               zoom_range=[1, 1],
                               horizontal_flip=False,
                               vertical_flip=False,
                               data_format='channels_last',
                               brightness_range=[1, 1])
        
        
        self.Xs = Xs
        self.ys = ys
        self.genImage = keras_image_generator
        self.batch_size = batch_size
        if   list_IDs is None:
            list_IDs = np.arange(ys.shape[0])
        labels = ys[list_IDs]
        
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.image_size = image_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y =  self.convertInputs(self.Xs[list_IDs_temp], self.ys[list_IDs_temp], self.image_size)
        g_x = self.genImage.flow(X , y,   batch_size=X.shape[0])

        return next(g_x)
   
    def resizeall(self, Xs, NN):
        
        Xs2 = np.zeros([Xs.shape[0], NN, NN, self.n_channels],dtype='f')
        for i in range(Xs.shape[0]):
            #if i % 1000 == 0:
                #print(i)
            xi = resize(Xs[i][:, :], (NN, NN)).reshape((NN, NN, self.n_channels))
            Xs2[i, : ,: ,0] = xi[: , :, 0]
            Xs2[i, : ,: ,1] = xi[: , :, 1]
            Xs2[i, : ,: ,2] = xi[: , :, 2]
        return Xs2
    
    def convertInputs(self, Xs , ys, image_size):
        if Xs.shape[1] != image_size:
            Xs =self.resizeall(Xs,image_size)
        ys = np_utils.to_categorical(ys, self.n_classes)
        return Xs, ys
   

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
     
        gc.collect()

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        print('""""""""""__data_generation""""""""""""')
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

class MultiOrientDataGenerator(DataGenerator):
    def __init__(self, Xs, ys, keras_image_generator=None ,  isFixedAngle = True, number_per_image=16, batch_size=32,
                 n_channels=3, n_classes=1, 
                 shuffle=True , image_size=128, list_IDs=None):
        super().__init__(  Xs, ys, keras_image_generator , batch_size,
                             n_channels, n_classes, 
                             shuffle , image_size, list_IDs)
        
        self.number_per_image = number_per_image
        #print(self.number_per_image, self.batch_size)
    
        self.rotate_image_generator =  ImageDataGenerator(rotation_range=180,
                               width_shift_range=0.0,
                               height_shift_range=0.0,
                               shear_range=0.00,
                               zoom_range=[1, 1],
                               horizontal_flip=False,
                               vertical_flip=False,
                               data_format='channels_last',
                              # fill_mode = 'constant',
                               brightness_range=[1, 1])
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)*self.number_per_image / self.batch_size))
            
    def __getitem__(self, index):
        'Generate one batch of data'
        image_index = int(round(self.batch_size/self.number_per_image, 0))
        
  
        indexes = self.indexes[index*image_index:(index+1)*image_index]
        # Find list of IDs
        #print("BBB", indexes)
        list_IDs_temp = [self.list_IDs[indexes[k//self.number_per_image]] for k in range(self.batch_size)]
       # print(list_IDs_temp, image_index, self.batch_size)
        
        X, y =  super().convertInputs(self.Xs[list_IDs_temp], self.ys[list_IDs_temp], self.image_size)
            
        g_x = self.genImage.flow(X , y,   batch_size=X.shape[0], shuffle=False)
        X2,y2 = next(g_x)
        gx_2 = self.rotate_image_generator.flow(X2 , y2,batch_size=X.shape[0],  shuffle=False)
        return next(gx_2)
        'Generate one batch of data'
        image_index = int(round(self.batch_size/self.number_per_image, 0))
        indexes = self.indexes[index*image_index:(index+1)*image_index]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[indexes[k//self.number_per_image]] for k in range(self.batch_size)]
        #print(list_IDs_temp, image_index, self.batch_size)
        
        X, y =  super().convertInputs(self.Xs[list_IDs_temp], self.ys[list_IDs_temp], self.image_size)
            
        g_x = self.genImage.flow(X , y,   batch_size=X.shape[0], shuffle=False)
        X2,y2 = next(g_x)
        gx_2 = self.rotate_image_generator.flow(X2 , y2,batch_size=X.shape[0],  shuffle=False)
        return next(gx_2)

class CutOutDataGenerator(DataGenerator):
    def __init__(self, Xs, ys, keras_image_generator=None ,  max_range_cut = 0.3, batch_size=32,
                 n_channels=3, n_classes=1, 
                 shuffle=True , image_size=128, list_IDs=None, replace_value = 0):
        super().__init__(  Xs, ys, keras_image_generator , batch_size,
                             n_channels, n_classes, 
                             shuffle , image_size, list_IDs)
        
        
        self.max_range_cut = max_range_cut
        self.replace_value = replace_value
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def apply_mask(self, image,  n_squares=1):
        h, w, channels = image.shape
        
        size_x = int(self.max_range_cut*self.image_size* (np.random.random() + 1)*0.5)
        size_y = int(self.max_range_cut*self.image_size* (np.random.random() + 1)*0.5)
        #print(size_x, size_y,h, w, channels)
        
        new_image = image
        for _ in range(n_squares):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - size_y // 2, 0, h)
            y2 = np.clip(y + size_y // 2, 0, h)
            x1 = np.clip(x - size_x // 2, 0, w)
            x2 = np.clip(x + size_x // 2, 0, w)
            new_image[y1:y2,x1:x2,:] = self.replace_value
        return new_image
            
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y =  super().convertInputs(self.Xs[list_IDs_temp], self.ys[list_IDs_temp], self.image_size)
        for i in range(X.shape[0]):
            X[i] = self.apply_mask(X[i])
        
        g_x = self.genImage.flow(X , y,   batch_size=X.shape[0])

        return next(g_x)

class MixupDataGenerator(DataGenerator):
    def __init__(self, Xs, ys, keras_image_generator=None , alpha=0.2, batch_size=32,
                 n_channels=3, n_classes=1, 
                 shuffle=True , image_size=128, list_IDs=None):
        super().__init__(  Xs, ys, keras_image_generator , batch_size,
                             n_channels, n_classes, 
                             shuffle , image_size, list_IDs)

        self.alpha = alpha

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def getListIDs(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return list_IDs_temp
    
    def __getitem__(self, index):
        'Generate one batch of data'
        
        l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = l.reshape(self.batch_size, 1, 1, 1)
        y_l = l.reshape(self.batch_size, 1)
        
        #print(l.shape, X_l.shape, y_l.shape)
        
        # Find list of IDs
        list_IDs_temp = self.getListIDs(index)
        X1, y1 =  super().convertInputs(self.Xs[list_IDs_temp], self.ys[list_IDs_temp], self.image_size)
        
        list_IDs_temp2 = self.getListIDs(np.random.randint(self.__len__()))
        X2, y2 =  super().convertInputs(self.Xs[list_IDs_temp2], self.ys[list_IDs_temp2], self.image_size)
        X = X1 * X_l + X2 * (1 - X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        
        g_x = self.genImage.flow(X , y,   batch_size=X.shape[0])

        return next(g_x)

print("Import Custom DataGenerator")