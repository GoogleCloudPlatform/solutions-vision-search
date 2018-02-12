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

import json
import logging
import os

from flask import current_app
from functools32 import lru_cache
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials
import resources

from google.appengine import runtime
from google.appengine.api import app_identity


@lru_cache()
def get_ml_svc():
  """Builds the ML Engine service object."""
  credentials = GoogleCredentials.get_application_default()
  ml_svc = discovery.build('ml', 'v1', credentials=credentials)
  return ml_svc


@lru_cache()
def read_category_map_as_json():
  """Reads the label to category mapping file."""
  cat_file_path = os.path.join(resources.__path__[0],
                               'category_map.json')
  with open(cat_file_path) as json_data:
    data = json.load(json_data)

  return data


def category_from_fixed_mappings(labels):
  """Gets the top category from fixed label to category mappings."""
  if not labels:
    return ''

  # Read label to category mappings from json file
  category_map = read_category_map_as_json()

  category_scores = {}
  top_category = None

  # Iterate through labels
  for label in labels:
    for k, v in category_map['categorymaps'].iteritems():
      if label['description'] in v:
        if k in category_scores.keys():
          category_scores[k] += label['score']
        else:
          category_scores[k] = label['score']

  num_category_scores = len(category_scores)
  if num_category_scores > 0:
    sorted_scores = sorted([(v, k) for (k, v) in category_scores.iteritems()],
                           reverse=True)
    category_tuple = sorted_scores[0]
    if category_tuple[0] > 0:
      top_category = category_tuple[1]

  # Return the top category name
  return top_category


def category_from_similar_vectors(labels):
  """Gets the most similar category by comparing image and category vectors."""
  if not labels:
    return []

  try:
    parent = 'projects/{}/models/{}/versions/{}'.format(
        app_identity.get_application_id(),
        current_app.config['ML_MODEL_NAME'],
        current_app.config['ML_MODEL_VERSION'])

    label_list = []
    score_list = []

    # Split labels with multiple words.
    # For simplicity, each word in the label is given the same score
    # as the original label.

    for label in labels:
      words = label['description'].split(' ')
      for word in words:
        label_list.append(word)
        score_list.append(label['score'])

    request_dict = {
        'instances': [{
            'labels': label_list,
            'scores': score_list
        }]
    }

    # Build and execute the request to ML Engine
    ml_svc = get_ml_svc()
    request = ml_svc.projects().predict(name=parent, body=request_dict)
    response = request.execute()

    # Return an array of similarity scores, one score for each category
    pred = response['predictions'][0]['prediction']
    return pred

  except runtime.DeadlineExceededError:
    logging.exception('Exceeded deadline in category_from_vector()')
