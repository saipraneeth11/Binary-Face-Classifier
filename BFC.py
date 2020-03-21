


from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten 
from keras.layers import Dense 

classifier = Sequential()
#building the model 
classifier.add(Convolution2D(32,3,3,input_shape = (64,64,3),activation = 'relu'))
# 64 is number of feature maps , 3,3 is the size of feature detector , we give input size of our image as input
classifier.add(Convolution2D(16,3,3,input_shape = (64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)) )
classifier.add(Flatten())
#now make the ANN
classifier.add(Dense(output_dim =128 , activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
#compile the model 

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics=['accuracy'])
#adam is like gradient desecnt where loss function is logisitic 

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'Train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'Ftest',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=1000,
        epochs=1,
        validation_data=test_set,
        nb_val_samples=1112)

#predicting a new image 
import numpy as np 
from skimage.transform import resize
import matplotlib.pyplot as plt
img = plt.imread('image10.jpg')
plt.imshow(img)
reimg = resize(img, (64,64,3)) 
img = plt.imshow(reimg) 
pred = classifier.predict(np.array( [reimg,] ))
print(pred)
