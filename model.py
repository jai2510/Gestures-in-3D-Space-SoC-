import numpy as np
import matplotlib.pyplot as plt
import keras

train_dir = "/Users/jai/hand_pointer/train"
validation_dir = "/Users/jai/hand_pointer/validation"
test_dir = "/Users/jai/hand_pointer/test"

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(rescale=1./255,
rotation_range=40, 
width_shift_range=0.2, 
height_shift_range=0.2, 
shear_range=0.2,
zoom_range=0.2, 
horizontal_flip=True, fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, 
                                              target_size=(150,150), 
                                              batch_size=15,
                                              class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(train_dir,
                                            target_size=(150,150), 
                                              batch_size=15,
                                              class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_dir,
                                              target_size=(150,150), 
                                              batch_size=15,
                                              class_mode='categorical')

from keras import models, layers, optimizers

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer=optimizers.RMSprop(lr=1e-4))

print("train or load ?")
ans = input()
if ans == "train":
    history = model.fit_generator(train_generator, 
                              steps_per_epoch=80, 
                              epochs=3, 
                              validation_data=validation_generator, 
                              validation_steps=10) 
    model.save("/Users/jai/hand_pointer/mypointermodel.h5")
if ans == "load":
    model = keras.models.load_model("/Users/jai/hand_pointer/mypointermodel.h5")
print(train_generator.class_indices)
