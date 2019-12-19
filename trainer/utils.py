from tensorflow.python.lib.io import file_io
import os
import numpy as np
from tensorflow.keras import backend as K
import zipfile
from .datagen import ImageDataGeneratorCustom
import logging


def downloads_training_images(data_path, is_cropped=False):
  pass 


class DataGenerator(object):
  def __init__(self, params, data_path, train_csv, val_csv, target_size=(224, 224)):
    self.params = params
    self.target_size = target_size
    self.idg = ImageDataGeneratorCustom(**params)
    self.data_path = data_path
    self.train_csv = train_csv
    self.val_csv = val_csv

  def get_train_generator(self, batch_size, is_full_data = False):
    with file_io.FileIO('dataset/' + self.train_csv, mode='r') as train_f:
      if is_full_data:
        with file_io.FileIO(self.data_path + self.train_csv, mode='r') as val_f:
          with file_io.FileIO(self.train_csv, mode='w+') as output_f:
            output_f.write(train_f.read()+"\n"+val_f.read())
      else:
        with file_io.FileIO(self.train_csv, mode='w+') as output_f:
          output_f.write(train_f.read().replace('\r', '').replace('\n', '').replace('"', '')[:-1])
    return self.idg.flow_from_directory("dataset/",
                                        batch_size = batch_size,
                                        target_size = self.target_size,shuffle=False,
                                        triplet_path =self.train_csv)

  def get_test_generator(self, batch_size):
    with file_io.FileIO('dataset/' +  self.val_csv, mode='r') as val_f:
      with file_io.FileIO(self.val_csv, mode='w+') as output_f:
        output_f.write(val_f.read().replace('\r', '').replace('\n', '').replace('"', '')[:-1])
    return self.idg.flow_from_directory("dataset/",
                                        batch_size = batch_size,
                                        target_size = self.target_size, shuffle=False,
                                        triplet_path = self.val_csv,
                                        should_transform = False)


def get_layers_output_by_name(model, layer_names):
    return {v: model.get_layer(v).output for v in layer_names}


def backup_file(job_dir, filepath):
  pass

def write_file_and_backup(content, job_dir, filepath):
    pass


def print_trainable_counts(model):
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    logging.info('Total params: {:,}'.format(trainable_count + non_trainable_count))
    logging.info('Trainable params: {:,}'.format(trainable_count))
    logging.info('Non-trainable params: {:,}'.format(non_trainable_count))

    return trainable_count, non_trainable_count