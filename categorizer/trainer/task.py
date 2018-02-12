# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import resources
from six import iteritems
from six.moves import urllib
import tensorflow as tf
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants

# Source URL for downloading GloVe embeddings
SOURCE_URL_PATH = 'http://nlp.stanford.edu/data/glove.6B.zip'


def maybe_download_and_extract(filename, data_dir, source_url):
  """Maybe download and extract a file."""
  if not gfile.Exists(data_dir):
    gfile.MakeDirs(data_dir)

  filepath = os.path.join(data_dir, filename)

  if not gfile.Exists(filepath):
    print('Downloading from {}'.format(source_url))
    temp_file_name, _ = urllib.request.urlretrieve(source_url)
    gfile.Copy(temp_file_name, filepath)
    with gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded \'{}\' of {} bytes'.format(filename, size))

  if filename.endswith('.zip'):
    print('Extracting {}'.format(filename))
    zipfile.ZipFile(file=filepath, mode='r').extractall(data_dir)


def read_categories_as_json():
  """Reads JSON file containing words to calculate category vectors."""
  cat_file_path = os.path.join(resources.__path__[0],
                               'categories.json')
  with open(cat_file_path) as json_data:
    data = json.load(json_data)
    num_categories = len(data['categories'])
    num_words_per_category = len(data['categories']['0'])
    category_words = np.empty([num_categories, num_words_per_category],
                              dtype='S48')

    for i in range(0, num_categories):
      for j in range(0, num_words_per_category):
        category_words[i][j] = data['categories'][str(i)][j]

  return category_words


def read_glove_embeddings(filename):
  """Read the GloVe embeddings and return vocab and embedding lists."""
  vocab, embed = [], []

  with open(filename, 'rb') as f:
    for line in f:
      tokens = line.decode('utf-8').rstrip().split()
      vocab.append(tokens[0])
      embed.append(tokens[1:])

  print('Size of vocabulary is {}'.format(len(vocab)))
  return vocab, embed


def get_category_embeddings(word_table, embeds):
  """Calculate embeddings from word labels for each category."""
  category_words = read_categories_as_json()
  word_ids = word_table.lookup(tf.constant(category_words))
  glove_embeds = tf.nn.embedding_lookup(embeds, word_ids)

  # Calculate category embedding by summing word vectors in each category
  # tf.reduce_sum is used as the category embedding will be normalized later
  category_embeds = tf.reduce_sum(glove_embeds, axis=1)
  expand_category_embeds = tf.expand_dims(category_embeds, axis=1)
  return expand_category_embeds


def get_label_embedding(labels, scores, word_table, embeds):
  """Calculate embeddings from word labels for each image."""
  word_ids = word_table.lookup(labels)
  glove_embeds = tf.nn.embedding_lookup(embeds, word_ids)
  normalized_scores = tf.divide(scores, tf.reduce_sum(scores))
  expanded_scores = tf.expand_dims(normalized_scores, axis=2)

  # Calculate linear combination of word vectors and scores
  # tf.reduce_sum is used as the label embedding will be normalized later
  labels_embed = tf.reduce_sum(tf.multiply(glove_embeds, expanded_scores),
                              axis=1)
  return labels_embed


def get_similarity(labels_embed, category_embeds):
  """Calculate the similarity between image and category embeddings."""
  labels_embed = tf.nn.l2_normalize(labels_embed, 1)
  category_embeds = tf.nn.l2_normalize(category_embeds, 2)
  cos_similarity = tf.reduce_sum(tf.multiply(labels_embed, category_embeds),
                                 axis=2)
  transpose_similarity = tf.transpose(cos_similarity)
  return transpose_similarity


def export_model(glove_filepath, model_dir, gcs_output_path):
  """Exports TensorFlow model."""
  vocab, embed = read_glove_embeddings(glove_filepath)

  # Add a zero vector for unknown words
  embed.insert(0, np.zeros(len(embed[0])).tolist())
  vocab.insert(0, '<UNK>')

  sess = tf.Session()

  with tf.Session(graph=tf.Graph()) as sess:
    # Store the GloVe embeddings
    embeddings = tf.Variable(tf.constant(np.array(embed).astype(np.float32),
                                         shape=[len(embed), len(embed[0])]),
                             trainable=False, name='embeddings')

    # Define a lookup table to convert word strings to ids
    # that correspond to the index positions of the list.
    word_table = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(vocab),
        num_oov_buckets=0,
        default_value=0)

    # Initialize global vars and word lookup table
    init_op = tf.global_variables_initializer()
    table_init_op = tf.tables_initializer()
    sess.run([init_op, table_init_op])

    # Get category embeddings and labels
    category_embeds = get_category_embeddings(word_table, embeddings)

    # Define prediction graph input placeholders
    labels_placeholder = tf.placeholder(tf.string, [None, None])
    scores_placeholder = tf.placeholder(tf.float32, [None, None])

    labels_embed = get_label_embedding(labels_placeholder,
                                       scores_placeholder,
                                       word_table,
                                       embeddings)
    similarity = get_similarity(labels_embed, category_embeds)

    inputs = {
        'labels': labels_placeholder,
        'scores': scores_placeholder
    }
    input_signatures = {}

    for key, val in iteritems(inputs):
      predict_input_tensor = meta_graph_pb2.TensorInfo()
      predict_input_tensor.name = val.name
      predict_input_tensor.dtype = val.dtype.as_datatype_enum
      input_signatures[key] = predict_input_tensor

    outputs = {'prediction': similarity}
    output_signatures = {}

    for key, val in iteritems(outputs):
      predict_output_tensor = meta_graph_pb2.TensorInfo()
      predict_output_tensor.name = val.name
      predict_output_tensor.dtype = val.dtype.as_datatype_enum
      output_signatures[key] = predict_output_tensor

    inputs_name, outputs_name = {}, {}

    for key, val in iteritems(inputs):
      inputs_name[key] = val.name
    for key, val in iteritems(outputs):
      outputs_name[key] = val.name

    tf.add_to_collection('inputs', json.dumps(inputs_name))
    tf.add_to_collection('outputs', json.dumps(outputs_name))

    predict_signature_def = signature_def_utils.build_signature_def(
        input_signatures, output_signatures,
        signature_constants.PREDICT_METHOD_NAME)
    build = builder.SavedModelBuilder(model_dir)
    build.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            predict_signature_def
        },
        legacy_init_op=tf.saved_model.main_op.main_op(),
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS))

    # Finally save the model
    build.save()

    # Copy to GCS
    if gcs_output_path:
      gcs_copy(model_dir, gcs_output_path)


def gcs_copy(source, dest):
  """Copies files to and from Google Cloud Storage."""
  print('Recursively copying from {} to {}'.format(source, dest))
  subprocess.check_call(['gsutil', '-q', '-m', 'cp', '-R']
                        + [source] + [dest])


def main(_):
  maybe_download_and_extract('glove.6B.zip', FLAGS.data_dir, SOURCE_URL_PATH)
  glove_filepath = os.path.join(FLAGS.data_dir, 'glove.6B.50d.txt')

  if gfile.Exists(glove_filepath):
    print('Exporting model to directory \'{}\''.format(FLAGS.model_dir))
    export_model(glove_filepath, FLAGS.model_dir, FLAGS.gcs_output_path)
  else:
    print('Could not find file \'{}\''.format(glove_filepath))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/model',
      help='Base directory for output models.'
  )

  parser.add_argument(
      '--data_dir',
      type=str,
      default='/tmp/data',
      help='Work directory for downloaded files.'
  )

  parser.add_argument(
      '--gcs_output_path',
      type=str,
      default=None,
      help='Google Cloud Storage output path.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
