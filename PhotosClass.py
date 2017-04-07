
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd

from matplotlib import cm
from matplotlib import colors
from scipy import interpolate

import random

import os
from tqdm import tnrange, tqdm_notebook

class Photos(object):
    def __init__(self, min_number = 0, max_number = 3000, root = "thumbnails_features_deduped_publish"):
        self.min_number = min_number
        self.max_number = max_number    
        self.min_number_tmp = min_number
        self.max_number_tmp = max_number 
        self.path = root      
        f = open('memo_amount.txt')
        self.amounts = pd.read_table(f,sep='\s+', header=None, names = ["Amounts"])
        self.numbers = np.int_(np.array(self.amounts))
 
            
    def fit(self):
        self.numbers_extended = []
        self.labels = []
        self.img = []
        self.shapes = []
        self.amounts = []
        self.numbers_extended_tmp = []
        self.labels_tmp = []
        self.img_tmp = []
        self.shapes_tmp = []
        self.amounts_tmp = []
      
        folders_chosen = np.array(os.listdir(path=self.path))[(((self.numbers>=self.min_number) * (self.numbers<=self.max_number)).reshape(1,-1)[0])] 
        for folder in tqdm_notebook(folders_chosen, desc = 'Folders'):
            folder_content = os.listdir(os.path.join(self.path,folder))
            if  folder_content[-1]=='Thumbs.db':
                length = len(os.listdir(os.path.join(self.path,folder)))-4
                images = os.listdir(os.path.join(self.path,folder))[:-4]
            else:
                length = len(os.listdir(os.path.join(self.path,folder)))-3
                images = os.listdir(os.path.join(self.path,folder))[:-3]
            for image in images: 
                self.labels_tmp.append(folder)
                img_tmp = mpimg.imread(os.path.join(self.path,folder, image))
                self.img_tmp.append(img_tmp)
                self.shapes_tmp.append(img_tmp.shape)
                self.numbers_extended_tmp.append(length)
        self.update()
       
        f.close()
        self.numbers_extended_tmp = np.array(self.numbers_extended_tmp)
        return self.img
        
    
    def get(self, min_number = 0, max_number = 3000):               
        self.min_number_tmp = min_number
        self.max_number_tmp = max_number
        changed_min = False
        changed_max = False
        
        min_set = [[],[], [], []]
        max_set = [[],[], [], []]
        if  (min_number < self.min_number):
            min_set = self.add_photos(min_number = self.min_number_tmp, max_number = self.min_number)
            self.min_number = min_number
            changed_min = True
        if  (max_number > self.max_number):
            max_set = self.add_photos(min_number = self.max_number, max_number = self.max_number_tmp)
            self.max_number = max_number            
            changed_max = True
            
        form = ((np.array(self.numbers_extended)>=self.min_number) * (np.array(self.numbers_extended)<=self.max_number))

        self.img_tmp=min_set[0]+list(np.array(self.img)[form])+max_set[0]
        self.labels_tmp=min_set[1]+list(np.array(self.labels)[form])+max_set[1]
        self.shapes_tmp=min_set[2]+list(np.array(self.shapes)[form])+max_set[2]
        self.numbers_extended_tmp=min_set[3]+list(np.array(self.numbers_extended)[form])+max_set[3]
        self.numbers_extended_tmp = np.array(self.numbers_extended_tmp)
        
        if changed_min or changed_max:
            self.update()
            
        return self.img_tmp
    
    def update(self):
            self.numbers_extended = self.numbers_extended_tmp
            self.labels  = self.labels_tmp
            self.img = self.img_tmp
            self.shapes = self.shapes_tmp
            self.amounts = self.amounts_tmp

    def add_photos(self, min_number = 0, max_number = 3000):
        images = []
        labels = []
        shapes = []
        numbers = []
        folders_chosen = np.array(os.listdir(path=self.path))[(((self.numbers>=min_number) * (self.numbers<=max_number)).reshape(1,-1)[0])]

        for folder in tqdm_notebook(folders_chosen, desc = 'Folders'):
            folder_content = os.listdir(os.path.join(self.path,folder))
            if  folder_content[-1]=='Thumbs.db':
                length = len(os.listdir(os.path.join(self.path,folder)))-4
                imgs = os.listdir(os.path.join(self.path,folder))[:-4]
            else:
                length = len(os.listdir(os.path.join(self.path,folder)))-3
                imgs = os.listdir(os.path.join(self.path,folder))[:-3]

            for image in imgs: 
                labels.append(folder)
                img_tmp = mpimg.imread(os.path.join(self.path,folder, image))
                images.append(img_tmp)
                shapes.append(img_tmp.shape)
                numbers.append(length)
        return [images, labels, shapes, numbers]
    
    def show(self, im = None, processed = False, numbers = None):     
        if im == None:
            if (numbers == None):
                numbers = range(len(self.img_tmp))
            elif (type(numbers) is not list):
                numbers = [numbers]
            if processed:
                images = self.img_pro_tmp
            else:
                images = self.img_tmp

            
            n = len(numbers)
            if n<4:
                b = n
                a = 1
            else:
                a = np.ceil(n/4)
                b = 4    
            fig = plt.figure(figsize = (a, b))
            for i, num in enumerate(numbers):
                ax = fig.add_subplot(a,b, i+1)
                plt.imshow(images[num])
                plt.title(self.labels_tmp[num])
                plt.xticks(())
                plt.yticks(())
        else: 
            fig = plt.figure(figsize = (15, 15))
            ax = fig.add_subplot(111)
            plt.imshow(im)
            plt.xticks(())
            plt.yticks(())
        plt.show()
        
   
    def preprocess(self):
        g = open('memo_shapes.txt')
        shapes = pd.read_table(g,sep=',\s+|\)|\(', header=None, names = ['H', 'h', 'v', 'RGB'], engine = 'python',
                      index_col = False).drop('H', axis=1)
        ideal_shape = [200, 200]
        self.img_pro_tmp = []
        for im in tqdm_notebook(self.img_tmp, desc = 'Preprocessing'):

            if len(im.shape) == 2:
                internal = [0]
            else: 
                internal = [np.array([0,0,0], dtype = 'uint8')]
            image_shape = im.shape
            a = im.shape
            dif = ideal_shape[0] - image_shape[0]
            high = (ideal_shape[0] - image_shape[0]+1)//2
            a1 = [high, dif-high, dif]
            if (high != 0):                
                new_image = np.append([np.repeat(internal,image_shape[1], axis = 0)]*high, im, axis = 0)
            if ((dif-high) != 0):
                new_image = np.append(new_image ,[np.repeat(internal,image_shape[1], axis = 0)]*(dif-high), axis = 0)
            if ((high != 0) or (dif - high) != 0):
                im = new_image

            dif = ideal_shape[1] - im.shape[1]
            high = (ideal_shape[1] - im.shape[1]+1)//2   
            b=im.shape
            b1 = [high, dif-high, dif]
            
            if (high != 0): 
                new_image = np.append([np.repeat(internal,high, axis = 0)]*im.shape[0], im, axis = 1)
            if ((dif-high) != 0):
                new_image = np.append(new_image, [np.repeat(internal,dif-high, axis = 0)]*im.shape[0], axis = 1)
            if ((high != 0) or (dif - high) != 0):
                im = new_image
            d=im.shape
            if len(d) != 3:
                print('original',a)
                print(a1)
                print('mid',b)
                print(b1)
                print('final',d)
            self.img_pro_tmp.append(im)
        g.close()
        return self.img_pro_tmp
    
    def get_some(self, processed = True, n = 20, amount = 20, full_random = False):
        self.numbers_extended_tmp = []
        self.labels_tmp = []
        self.img_tmp = []
        self.shapes_tmp = []
        self.amounts_tmp = []
        if full_random == False:
            folders_chosen = list(np.array(os.listdir(path=self.path))[((self.numbers>=amount).reshape(1,-1)[0])])
            folders_chosen = [i for i in random.sample(folders_chosen, n)]
            for folder in tqdm_notebook(folders_chosen, desc = 'Uploading'):
                folder_content = os.listdir(os.path.join(self.path,folder))
                if  folder_content[-1]=='Thumbs.db':
                    length = len(os.listdir(os.path.join(self.path,folder)))-4
                    images = os.listdir(os.path.join(self.path,folder))[:-4]
                else:
                    length = len(os.listdir(os.path.join(self.path,folder)))-3
                    images = os.listdir(os.path.join(self.path,folder))[:-3]
                images = [i for i in random.sample(images, amount)]
                for image in images: 
                    self.labels_tmp.append(folder)
                    img_tmp = mpimg.imread(os.path.join(self.path,folder, image))
                    self.img_tmp.append(img_tmp)
                    self.shapes_tmp.append(img_tmp.shape)
                    self.numbers_extended_tmp.append(length)
        else:
            e = open('memo_labels.txt')
            labels = pd.read_table(e,sep='\n', header=None, names = ['L'])
            labels = list(labels['L'])
            actors_chosen = [i for i in random.sample(labels, n)]
            for folder in tqdm_notebook(actors_chosen, desc = 'Photos'):
                folder_content = os.listdir(os.path.join(self.path,folder))
                if  folder_content[-1]=='Thumbs.db':
                    length = len(os.listdir(os.path.join(self.path,folder)))-4
                    images = os.listdir(os.path.join(self.path,folder))[:-4]
                else:
                    length = len(os.listdir(os.path.join(self.path,folder)))-3
                    images = os.listdir(os.path.join(self.path,folder))[:-3]
                image = random.sample(images, 1)[0]
                self.labels_tmp.append(folder)
                img_tmp = mpimg.imread(os.path.join(self.path,folder, image))
                self.img_tmp.append(img_tmp)
                self.shapes_tmp.append(img_tmp.shape)
                self.numbers_extended_tmp.append(length)
        if processed:
            return self.preprocess(), self.labels_tmp
        else:
            return self.img_tmp, self.labels_tmp
        

