# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 16:53:48 2019

@author: Ali_Asaf
"""
"""
*Bu Kurulumların Yapılması Gerekmektedir.*
pip install tensorflow
pip install keras
pip install matplotlib
pip install pillow
pip install opencv-python
"""

from tensorflow.keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import keras
from keras.preprocessing import image
from keras import Sequential
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers.core import Dense,Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy,sparse_categorical_crossentropy
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
import itertools
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import load_model
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2
import glob
import pickle
from scipy.spatial import distance


class img_vektor:               # Vektör hesabu yaptıktan sonra resim ile vektörünü tutsun.
    
    def __init__(self, path, vektor):
        self.path = path
        self.vektor = vektor
    

def vektor_bul(model,layer_name,resim): # Verilen katmana göre resmin vektörünü bulup döndürür.
  
  layer_output=model.get_layer(layer_name).output           # Verilen modelin ilgili katmanını bul.
 
  intermediate_model=keras.models.Model(inputs=model.input,outputs=layer_output) # Katmana göre yeni model oluştur.

  intermediate_prediction=intermediate_model.predict(resim) # Çıktısını tahmin et.
  
  return intermediate_prediction                            # Vektörü döndür.

def pickled_items(filename):                                # Verilen filedan file sonuna kodar okuma yapar.
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


train_path='E:\\image_database\\proje\\train'				# Resimlerin bu şekilde pathlerinin verilmesi gerekir.
valid_path='E:\\image_database\\proje\\validation'		
test_path='E:\\image_database\\proje\\test'

train_batches=ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224),classes=['all_souls','ashmolean','balliol','bodleian','christ_church',
                                 'cornmarket','hertford','jesus','keble','magdelen','new','oriel','oxford','pitt_rivers','redcliffe_camera',
                                 'trinity','worcester'],batch_size=128)
valid_batches=ImageDataGenerator().flow_from_directory(valid_path,target_size=(224,224),classes=['all_souls','ashmolean','balliol','bodleian','christ_church',
                                 'cornmarket','hertford','jesus','keble','magdelen','new','oriel','oxford','pitt_rivers','redcliffe_camera',
                                 'trinity','worcester'],batch_size=128)
test_batches=ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224),classes=['all_souls','ashmolean','balliol','bodleian','christ_church',
                                 'cornmarket','hertford','jesus','keble','magdelen','new','oriel','oxford','pitt_rivers','redcliffe_camera',
                                 'trinity','worcester'],batch_size=128)


model = Sequential()
model.add(Conv2D(16,(3,3),padding='valid',input_shape=(224,224,3),name='con_1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(16,(3,3),padding='valid',name='con_2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='valid',data_format=None))
model.add(Conv2D(16,(3,3),padding='valid',activation='relu',name='con_3'))
model.add(Flatten())
model.add(Dense(512,activation='relu',name='dense_1'))
model.add(Dense(17,activation='softmax',name='dense_2'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#visualize_conv_layer(model,'dense_1',train_batches)
#model.fit(test_images,test_labels)
history=model.fit_generator(train_batches,steps_per_epoch=30,validation_data=valid_batches,validation_steps=30,epochs=5,verbose=1)

model.save('modelim_filtre16_V2.h5')
model.save_weights('modelim_filtre16_weight_v2s1.h5')

score=model.evaluate(test_batches, verbose=1)
print(score)


#Çıkan sonuçları grafik olarak göster.
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


with open('train_degerleri_V2.txt', 'wb') as output:
    for dosya in glob.glob(train_path+"/*"):
       for resim in glob.glob(dosya+"/*"):
           img_getir=cv2.imread(resim)
           img_getir=cv2.resize(img_getir,(224,224))
           img_getir=img_getir.reshape(1,224,224,3)
           sonuc=vektor_bul(model,'dense_1',img_getir)
           resim_vektoru=img_vektor(resim,sonuc)
           pickle.dump(resim_vektoru, output, pickle.HIGHEST_PROTOCOL)

with open('train_degerleri_V2.txt', 'ab') as output:
    for dosya in glob.glob(test_path+"/*"):
       for resim in glob.glob(dosya+"/*"):
           img_getir=cv2.imread(resim)
           img_getir=cv2.resize(img_getir,(224,224))
           img_getir=img_getir.reshape(1,224,224,3)
           sonuc=vektor_bul(model,'dense_1',img_getir)
           resim_vektoru=img_vektor(resim,sonuc)
           pickle.dump(resim_vektoru, output, pickle.HIGHEST_PROTOCOL)
  

flag=0      # Belirli sayıda çıktı göstermek için.
count=0     # Hangi aralıkları görmek istiyorsak
for dosya in glob.glob(test_path+"/*"):
       for resim in glob.glob(dosya+"/*"):
           resim_oklid = []
           img_getir=cv2.imread(resim)                  # Verilen pathden resimi al.
           img_getir=cv2.resize(img_getir,(224,224))    # Boyutunu ayarla.
           img_getir=img_getir.reshape(1,224,224,3)     
           sonuc=vektor_bul(model,'dense_1',img_getir)  # Verilen resmin dense_1 katmanındaki sonucunu al.
           resim_vektoru=img_vektor(resim,sonuc)        # Sonuç ile resmin pathni tut.
           for img in pickled_items('train_degerleri_V2.txt'):
               dst = distance.euclidean(sonuc, img.vektor) # Kaydedilen değerler ile karşılaştır.
               oklid_vektor=img_vektor(img.path,dst)    # Öklid mesafelerini ve pathini kaydet.
               if(img.path!=resim):                     # Resim kendisiyle eşleşmesin.
                   resim_oklid.append(oklid_vektor)     # Listeye ekle.
           
           for k in range(0,len(resim_oklid)):          # Hesaplanan uzaklık değerlerini sırala.
               for s in range(0, len(resim_oklid) - k - 1):
                   if resim_oklid[s].vektor > resim_oklid[s + 1].vektor:
                       resim_oklid[s], resim_oklid[s + 1] = resim_oklid[s + 1], resim_oklid[s] 
           
           if(flag):
                for i in range(0,5):
                    print("Orjinal Resim : ",resim,"\nBenzetilen resim ",i," : ",resim_oklid[i].path)
                    res=cv2.imread(resim,1)             #resimleri göster.  
                    show=cv2.imread(resim_oklid[i].path)
                    plt.imshow(res)
                    plt.show()
                    plt.imshow(show)
                    plt.show()
                    if(count == 5):#sıfırdan  e kadar olan resimleri gösterdiği için 5 olunca göstermeyi durdursun.
                        flag=0
           count=count+1
           if(count == 1):      # Belirli aralıklardaki resmi göstermek istersek örneğin 10-15 bu değeri 10 içeridekini 15 yapıyoruz.
               flag=1    
                    



#---------------VGG16-------------------
from tensorflow.keras.optimizers import SGD
vgg16_model=keras.applications.vgg16.VGG16() #VGG yi al.

vgg16_model.layers.pop()                    # Son layeri çıkar.

x = vgg16_model.layers[-1].output           # Son layere kadar olan çıktıları al.
x = Dense(17, activation='softmax', name='predictions_me')(x) # Kendi katmanımı koy.
yeni_model = Model(input=vgg16_model.input,output=x) # Modeli elde et.

for layer in yeni_model.layers[:-3]:        # Dense layere kadar olan kısımların trainable değerlerini kapat.
    layer.trainable=False
    #print(layer.name)
        

for layer in yeni_model.layers[-3:]:        # Dense layerların trainable değerlerini aç.
    layer.trainable=True
    #print(layer.name)    
    
yeni_model.summary()                        # Modeli göster.

# Modeli eğit.    
yeni_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
vgg_history=yeni_model.fit_generator(train_batches,steps_per_epoch=5,validation_data=valid_batches,validation_steps=5,epochs=75,verbose=1)

#Modeli kaydet.
model.save('modelim_vgg.h5')
model.save_weights('modelim_vgg_weights1.h5')

# Test değeriyle test et.
score=yeni_model.evaluate(test_batches, verbose=1)
print(score)

# Sonuçları grafik olarak göster.
plt.plot(vgg_history.history['accuracy'])
plt.plot(vgg_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(vgg_history.history['loss']) 
plt.plot(vgg_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()