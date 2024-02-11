import tensorflow
import os
import shutil
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import joblib

#Importing custom model
from model import ts_model

#tf.keras.preprocessing.image_dataset_from_directory
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def copy_images_to_dir(path,images_to_copy, destination):
    for image in images_to_copy:
        shutil.copyfile(f'{path}{image}', f'{destination}/{image}')

def distribute_train_validation_split(validation_size=0.2):
    file_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    no_graffiti = os.listdir(file_dir + '\\TrainFinal negative\\')
    yes_graffiti = os.listdir(file_dir + '\\TrainFinal positive\\')

    #Defining index's for split
    index_to_split_no = int(len(no_graffiti) * validation_size)
    index_to_split_yes = int(len(yes_graffiti) * validation_size)

    #Splitting data into test and train
    training_graffiti = yes_graffiti[index_to_split_yes:]
    validation_graffiti = yes_graffiti[:index_to_split_yes]

    training_no_graffiti = no_graffiti[index_to_split_no:]
    validation_no_graffiti = no_graffiti[:index_to_split_no]

    #shutil.rmtree(file_dir + "\\Model\\input_for_model")
    os.makedirs('./Development/Model/input_for_model/train/graffiti/', exist_ok=True)
    os.makedirs('./Development/Model/input_for_model/train/no_graffiti/', exist_ok=True)
    os.makedirs('./Development/Model/input_for_model/validation/graffiti/', exist_ok=True)
    os.makedirs('./Development/Model/input_for_model/validation/no_graffiti/', exist_ok=True)

    try:
        copy_images_to_dir('./Development/TrainFinal positive/',
            training_graffiti, './Development/Model/input_for_model/train/graffiti')
    except:
        print("wrong index")
    try:
        copy_images_to_dir('./Development/TrainFinal positive/',
            validation_graffiti, './Development/Model/input_for_model/validation/graffiti')
    except:
        print("wrong index")

    try:
        copy_images_to_dir('./Development/TrainFinal negative/',
            training_no_graffiti, './Development/Model/input_for_model/train/no_graffiti')
    except:
        print("wrong index")

    try:
        copy_images_to_dir('./Development/TrainFinal negative/',
            validation_no_graffiti, './Development/Model/input_for_model/validation/no_graffiti')
    except:
        print("wrong index")

def plot_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def plot_train(history):
    acc = history.history['BinaryAccuracy']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.legend()
    plt.figure()

    loss = history.history['loss']
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()



if __name__ =="__main__":


    train_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)
    validation_imagedatagenerator = ImageDataGenerator(rescale=1/255.0)
    
    train_iterator = train_imagedatagenerator.flow_from_directory(
        './Development/Model/input_for_model/train',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_iterator = validation_imagedatagenerator.flow_from_directory(
        './Development/Model/input_for_model/validation',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    #Creating model
    my_model = ts_model()
    model = my_model.model


    #Training model
    history = model.fit(train_iterator,
                    validation_data=validation_iterator,
                    steps_per_epoch=20,
                    epochs=10)
    # Save the weights
    #model.save_weights('./Development/Model/model_weights/')
    # Save the entire model as a `.keras` zip archive.
    #model.save('./Development/Model/model_weights/my_model.h5',save_format='h5')
    joblib.dump(model, 'my_model.pkl')
    
    #Visualizing results
    print("HISTORY")
    print(history.history)

    batch_size = 20
    predictions = model.predict_generator(validation_iterator)

    #Structuring predicted/actual values
    y_true = validation_iterator.classes
    y_pred = np.array([np.argmax(x) for x in predictions])

    #Showing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    #Writing output
    validation_compare = pd.DataFrame({'y_true': y_true,
                                        'y_pred': y_pred})
    print(validation_compare)
    
    validation_compare['Correct'] = np.where(validation_compare['y_true'] == validation_compare['y_true'], 1, 0)
    validation_compare.to_csv('./Development/Model/performance/Validation_Performance.csv')

    history_df = pd.DataFrame.from_dict(history.history)
    history_df.to_csv('./Development/Model/performance/history.csv')
    
    

