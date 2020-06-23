from utils.DataGenerator import *
from skimage import transform

class MultiOrientTileDataGenerator(DataGenerator):
    def __init__(self, Xs, ys, keras_image_generator=None ,  dim = 4, batch_size=32,
                 n_channels=3, n_classes=1, 
                 shuffle=True , image_size=128, list_IDs=None):
        
        super().__init__(  Xs, ys, keras_image_generator , batch_size,
                             n_channels, n_classes, 
                             shuffle , image_size, list_IDs)
        
        self.angle = 360//(dim*dim)
        self.nTiles = dim*dim
        self.dim =dim

        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
            
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[indexes[k]] for k in range(self.batch_size)]
        #print(list_IDs_temp, image_index, self.batch_size)
        
        Xs, ys =  self.Xs[list_IDs_temp], self.ys[list_IDs_temp]
        #print(Xs.shape, ys.shape, Xs.shape[0]*self.nTiles, 360//self.angle, self.angle)
        multiOrientGen = MultiOrientDataGenerator(Xs, ys,
                                                      number_per_image = self.nTiles, 
                                                      batch_size = self.nTiles, 
                                                      n_channels=self.n_channels,
                                                      n_classes = self.n_classes,shuffle=False,  
                                                      image_size= int(round(self.image_size//self.dim)))
        #print('AAA', multiOrientGen.indexes)
        #print(Xs.shape[0])
        #print(multiOrientGen.number_per_image, 360//16)
        for i in range(Xs.shape[0]):
            Xs_c, ys_c = multiOrientGen.__getitem__(i)
            _, w, h, c = Xs_c.shape
            #print(i,w,h)
            img = np.zeros((w*self.dim, h*self.dim, c)) 
            for ii in range(self.dim):
                for jj in range(self.dim):
                    kk = ii*self.dim + jj
                    img[ii*w: (ii+1)*w ,jj*h: (jj+1)*h , :] = Xs_c[kk]
            Xs[i] = transform.resize(img, (Xs[i].shape[0], Xs[i].shape[1]))
            
        
            
        g_x = self.genImage.flow(Xs , ys,   batch_size=Xs.shape[0], shuffle=False)
        return Xs, np_utils.to_categorical(ys, self.n_classes)

class CustomMixDataGenerator(DataGenerator):
    def __init__(self, Xs, ys, keras_image_generator=None , 
                 configDic = {'isMutiOrientationTiles' : True,
                              'isCutout' : True,
                              'isMixedup':True,
                              'dim_tile': 4,
                              'range_cut': 0.4,
                              'mixup_alpha':0.3,
                              'replace_cut_value':0
                             },
                 batch_size=32,
                 n_channels=3, n_classes=1, 
                 shuffle=True , image_size=128, list_IDs=None):
        
        super().__init__(  Xs, ys, keras_image_generator , batch_size,
                             n_channels, n_classes, 
                             shuffle , image_size, list_IDs)
        
        self.configDic = configDic
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def getListIDs(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return list_IDs_temp
    
    def apply_mask(self, image,  n_squares=1):
        h, w, channels = image.shape
        
        max_range_cut = self.configDic['range_cut']
        size_x = int(max_range_cut*self.image_size* (np.random.random() + 1)*0.5)
        size_y = int(max_range_cut*self.image_size* (np.random.random() + 1)*0.5)
        #print(size_x, size_y,h, w, channels)
        
        new_image = image
        for _ in range(n_squares):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - size_y // 2, 0, h)
            y2 = np.clip(y + size_y // 2, 0, h)
            x1 = np.clip(x - size_x // 2, 0, w)
            x2 = np.clip(x + size_x // 2, 0, w)
            new_image[y1:y2,x1:x2,:] = self.configDic['replace_cut_value']
        return new_image
            
    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[indexes[k]] for k in range(self.batch_size)]
        #print(list_IDs_temp, image_index, self.batch_size)
        
        Xs, ys =  super().convertInputs(self.Xs[list_IDs_temp], self.ys[list_IDs_temp], self.image_size)
        
        if  self.configDic['isMixedup']:
            alpha = self.configDic['mixup_alpha']
            l = np.random.beta(alpha, alpha, self.batch_size)
            X_l = l.reshape(self.batch_size, 1, 1, 1)
            y_l = l.reshape(self.batch_size, 1)
            
            list_IDs_temp2 = self.getListIDs(np.random.randint(self.__len__()))
            X2, y2 =  super().convertInputs(self.Xs[list_IDs_temp2], self.ys[list_IDs_temp2], self.image_size)
            Xs = Xs * X_l + X2 * (1 - X_l)
            ys = ys * y_l + y2 * (1 - y_l)
            
        if  self.configDic['isCutout']:
            for i in range(Xs.shape[0]):
                Xs[i] = self.apply_mask(Xs[i])
                
        if  self.configDic['isMutiOrientationTiles']:
            multiOrientTileGen = MultiOrientTileDataGenerator(Xs, self.ys[list_IDs_temp], None, 
                                                          image_size=self.image_size, 
                                                          dim = self.configDic['dim_tile'], 
                                                          n_classes=self.n_classes, batch_size=self.batch_size)
            Xs, ys = multiOrientTileGen.__getitem__(0)

        g_x = self.genImage.flow(Xs , ys,   batch_size=Xs.shape[0])
        
        return next(g_x)
