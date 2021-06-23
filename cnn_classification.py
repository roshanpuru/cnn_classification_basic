import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

base_dir = './data/'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

"""
    Pre Process the data: Image to Numeric and resizing all input data to 150x150
"""


######################################## PRE-PROCESSING #########################################

def dataPreProcessing():
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20,
                                                        class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20,
                                                            class_mode='binary')
    return train_generator, validation_generator


######################################## MODEL #########################################
"""
Size: 150x150
Pooled till: 7x7
Feature Maps: 32(148x148) ----> 128(7x7)
Dense Layer: 1 --> Sigmoid -- Binary Classification
"""


def build_model():
    model = models.Sequential()

    # 32 Features of 148x148 each
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    # Output reduced to 74x74
    model.add(layers.MaxPooling2D((2, 2)))

    # 64 Features of 72x72 each
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # Output reduced to 36x36
    model.add(layers.MaxPooling2D((2, 2)))

    # 128 Features of 34x34 each
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # Output reduced to 17x17
    model.add(layers.MaxPooling2D((2, 2)))

    # 128 Features of 15x15 each
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # Output reduced to 7x7
    model.add(layers.MaxPooling2D((2, 2)))

    # 1. Flatten -- 6272 2. 512 filter vector 3. Single sigmoid output
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


################################# Optional - Visualisation #####################
def visualize():
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


######################################         ####################################
"""
Load the image and convert into image tensor to be submitted to model for prediction
"""


def load_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


####################################### Steps #####################################

train_generator, validation_generator = dataPreProcessing()
model = build_model()
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
model.save('cnn_binary_classification.h5')

#Load and Convert the image and make prediction
cat_img_path = './data/test/cats/cat.1501.jpg'
dog_img_path = './data/test/dogs/dog.1501.jpg'

cat_img = load_image(cat_img_path)
dog_img = load_image(dog_img_path)

print(model.predict(cat_img))
print(model.predict(dog_img))

# Optional - For Development and Visualisation purposes
# visualize()
