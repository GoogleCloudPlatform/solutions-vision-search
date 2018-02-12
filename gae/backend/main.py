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

import base64
import json
import logging
import os

from category_scorer import category_from_fixed_mappings
from category_scorer import category_from_similar_vectors
from flask import current_app, Flask, request, abort, jsonify, make_response
from flask_hashing import Hashing
from functools32 import lru_cache
from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

from google.appengine.api import images
from google.appengine.api import search

app = Flask(__name__)

hashing = Hashing(app)

# Configure the following envrionment variables via app.yaml

app.config['MAX_SEARCH_RESULTS'] = os.environ['MAX_SEARCH_RESULTS']
app.config['USE_CATEGORY_PREDICTOR'] = os.environ['USE_CATEGORY_PREDICTOR']
app.config['THUMB_SIZE'] = os.environ['THUMB_SIZE']
app.config['PREVIEW_SIZE'] = os.environ['PREVIEW_SIZE']
app.config['ML_MODEL_NAME'] = os.environ['ML_MODEL_NAME']
app.config['ML_MODEL_VERSION'] = os.environ['ML_MODEL_VERSION']
app.config['ML_CATEGORY_KEYS'] = os.environ['ML_CATEGORY_KEYS'].split(',')


@lru_cache()
def get_vision_svc():
  """Builds the Vision API service object."""
  credentials = GoogleCredentials.get_application_default()
  vision_svc = discovery.build('vision', 'v1', credentials=credentials)
  return vision_svc


@app.route('/_ah/push-handlers/image', methods=['POST'])
def pubsub_push():
  """Processes Pub/Sub notification triggered from image upload."""
  # Get the details of the GCS object encapsulated within the Pub/Sub message
  envelope = json.loads(request.data.decode('utf-8'))
  data = envelope['message']['data']
  attributes = envelope['message']['attributes']
  user_metadata = {}

  bucket_id = attributes['bucketId']
  object_id = attributes['objectId']
  event_type = attributes['eventType']
  payload_format = attributes['payloadFormat']

  # Process when new object arrives or is overwritten in bucket
  if event_type == 'OBJECT_FINALIZE':

    # Get object metadata to add to the search index if present
    if payload_format == 'JSON_API_V1':
      object_metadata = json.loads(base64.b64decode(data))
      if 'metadata' in object_metadata:
        user_metadata = object_metadata['metadata']

    mapped_category = ''
    most_similar_category = ''

    # Request label detection on the uploaded object
    labels = detect_labels(bucket_id, object_id)
    num_detected_labels = len(labels)

    if num_detected_labels > 0:
      # Get the top category from fixed category mappings
      mapped_category = category_from_fixed_mappings(labels)

      if current_app.config['USE_CATEGORY_PREDICTOR'] == 'True':
        # Use ML Engine to calculate and return most similar category
        category_scores = category_from_similar_vectors(labels)
        max_similarity = max(category_scores)
        index = category_scores.index(max_similarity)
        most_similar_category = current_app.config['ML_CATEGORY_KEYS'][index]

    # Get thumbnail url
    thumb_size = int(current_app.config['THUMB_SIZE'])
    thumb_url = get_image_url(bucket_id,
                              object_id,
                              thumb_size)

    # Get preview url
    preview_size = int(current_app.config['PREVIEW_SIZE'])
    preview_url = get_image_url(bucket_id,
                                object_id,
                                preview_size)

    # Add labels to the search index
    doc_id = add_to_search_index(object_id,
                                 labels,
                                 user_metadata,
                                 mapped_category,
                                 most_similar_category,
                                 thumb_url,
                                 preview_url)

    logging.info('Document ID added: %s', doc_id)

  return 'OK', 200


@app.route('/query', methods=['GET'])
def query():
  """Queries the search index for matching documents."""
  response_dict = {}

  # Get query string
  query_text = request.args.get('text', '')

  # Get facet refinements
  query_refine = request.args.get('refine', None)
  facet_refinements = []
  if query_refine:
    facet_refinements = query_refine.split(',')

  # Set search options
  max_search_results = int(current_app.config['MAX_SEARCH_RESULTS'])
  search_options = search.QueryOptions(limit=max_search_results)

  # Execute the search
  index = search.Index(name='imagesearch')
  search_query = search.Query(query_text,
                              return_facets=['label_facet',
                                             'mapped_category_facet',
                                             'most_similar_category_facet'],
                              facet_refinements=facet_refinements,
                              options=search_options)

  search_results = index.search(search_query)

  # Iterate through matching documents and add to response
  matched_docs = []
  for result in search_results.results:

    # Iterate and gather index fields for matching document
    fields = {}
    for field in result.fields:
      if field.name in fields:
        fields[field.name].append(field.value)
      else:
        fields[field.name] = [field.value]

    # Add matching document details to list of documents
    matched_docs.append({
        'doc_id': result.doc_id,
        'rank': result.rank,
        'fields': fields,
    })

  # Iterate through matching facets and add to response
  matched_facets = {}
  for facet in search_results.facets:
    matched_facets[facet.name] = []
    for value in facet.values:
      matched_facets[facet.name].append({
          'facet_value': value.label,
          'facet_count': value.count,
          'facet_token': value.refinement_token
      })

  # Build and return response document
  num_matched_docs = len(matched_docs)
  if num_matched_docs > 0:
    response_dict['documents'] = matched_docs
  num_matched_facets = len(matched_facets)
  if num_matched_facets > 0:
    response_dict['facets'] = matched_facets

  response_dict['result'] = 'ok'

  resp = make_response(jsonify(response_dict))
  resp.headers['Access-Control-Allow-Origin'] = '*'

  return resp


@app.route('/delete', methods=['GET'])
def delete():
  """Deletes a document from the index."""
  try:
    response_dict = {}
    doc_id = request.args.get('id')
    search.Index(name='imagesearch').delete([doc_id])
    response_dict['result'] = 'ok'

  except search.DeleteError:
    logging.exception('Something went wrong in delete()')

  return jsonify(response_dict)


def detect_labels(bucket_id, object_id):
  """Detects labels from uploaded image."""
  try:
    # Construct GCS uri path
    gcs_image_uri = 'gs://{}/{}'.format(bucket_id, object_id)

    # Build request payload dict for label detection
    request_dict = [{
        'image': {
            'source': {
                'gcsImageUri': gcs_image_uri
            }
        },
        'features': [{
            'type': 'LABEL_DETECTION',
            'maxResults': 10,
        }]
    }]

    vision_svc = get_vision_svc()
    api_request = vision_svc.images().annotate(body={
        'requests': request_dict
    })
    response = api_request.execute()
    labels = []

    if 'labelAnnotations' in response['responses'][0]:
      labels = response['responses'][0]['labelAnnotations']

    return labels

  except runtime.DeadlineExceededError:
    logging.exception('Exceeded deadline in detect_labels()')


def add_to_search_index(object_id, labels, metadata, mapped_category=None,
                        most_similar_category=None, thumb_url=None,
                        preview_url=None):
  """Adds document to the search index."""
  try:
    # Define document search fields - these can be queried using keyword search
    fields = [
        search.TextField(name='image_id', value=object_id),
    ]

    # Define document facet fields
    facets = []

    # Add label descriptions into search and facet fields. Search API allows
    # multiple values for the same field.
    for label in labels:
      fields.append(search.TextField(name='label',
                                     value=label['description']))
      facets.append(search.AtomFacet(name='label_facet',
                                     value=label['description']))

    # Add mapped category and most similar category as facets
    if mapped_category:
      fields.append(search.TextField(name='mapped_category',
                                     value=mapped_category))
      facets.append(search.AtomFacet(name='mapped_category_facet',
                                     value=mapped_category))

    if most_similar_category:
      fields.append(search.TextField(name='most_similar_category',
                                     value=most_similar_category))
      facets.append(search.AtomFacet(name='most_similar_category_facet',
                                     value=most_similar_category))

    # We're not using a database, so store the image URLs to the index.
    # We don't need to search on the image URL, but we will need them to display
    # images in the user interface.

    # Add thumbnail url
    if thumb_url:
      fields.append(search.TextField(name='thumb_url',
                                     value=thumb_url))

    # Add preview url
    if thumb_url:
      fields.append(search.TextField(name='preview_url',
                                     value=preview_url))

    # Add any other object metadata as document search fields
    for k, v in metadata.iteritems():
      fields.append(search.TextField(name=k, value=v))

    # Add the document to the search index
    d = search.Document(doc_id=hashing.hash_value(object_id),
                        fields=fields,
                        facets=facets)
    add_result = search.Index(name='imagesearch').put(d)
    doc_id = add_result[0].id

    return doc_id

  except search.Error:
    logging.exception('Something went wrong in add_to_search_index()')


def get_image_url(bucket_id, object_id, size):
  """Gets a serving URL for an uploaded image."""
  try:
    url = ''
    filename = '/gs/{}/{}'.format(bucket_id, object_id)
    url = images.get_serving_url(None,
                                 filename=filename,
                                 secure_url=True,
                                 size=size)
    return url

  except images.Error:
    logging.exception('Something went wrong in get_image_url()')


@app.errorhandler(500)
def server_error(e):
  logging.exception('An error occurred during a request.')
  return """
  An internal error occurred: <pre>{}</pre>
  See logs for full stacktrace.
  """.format(e), 500
