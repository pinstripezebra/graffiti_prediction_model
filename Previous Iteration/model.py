import keras
import tensorflow_addons as tfa

class ts_model:

    def __init__(self):
        self.model = self.create_model()


    def create_model(self):

        # Defining data augmentation layer
        data_augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.2),
            ])
        #Defining model
        model = keras.Sequential([
            data_augmentation,
            keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPool2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        #Compling model
        model.compile(loss='binary_crossentropy',
                        optimizer='adam', 
                        metrics=[tfa.metrics.F1Score(num_classes = 2, threshold = 0.5, average="micro")])

        return model
        