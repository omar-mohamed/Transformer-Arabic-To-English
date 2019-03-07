from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
import time

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import translate
from utils import tokenizer
import transformer_main

_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6

# load model

tf.logging.set_verbosity(tf.logging.INFO)
subtokenizer = tokenizer.Subtokenizer("./tiny-model/vocab.ende.32768")

params = transformer_main.PARAMS_MAP["tiny"]
params["beam_size"] = _BEAM_SIZE
params["alpha"] = _ALPHA
params["extra_decode_length"] = _EXTRA_DECODE_LENGTH
params["batch_size"] = _DECODE_BATCH_SIZE
estimator = tf.estimator.Estimator(
    model_fn=transformer_main.model_fn, model_dir="./tiny-model/",
    params=params)

# input_data = "حبيبي يا عاشق"
# tf.logging.info("Translating text: %s" % input_data)
# start = time.time()
# print("started timing")
# output_data = translate.translate_text(estimator, subtokenizer, input_data)
# end = time.time()
# print("translate took %f seconds" % (end - start))


# define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default

# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)


# API route
@app.route('/api', methods=['POST'])
def api():
    input_data = request.json
    app.logger.info("api_input: " + str(input_data))
    tf.logging.info("Translating text: %s" % input_data)
    output_data = translate.translate_text(estimator, subtokenizer, input_data)
    app.logger.info("api_output: " + str(output_data))

    response = jsonify(output_data)
    return response


@app.route('/')
def index():
    return "Index API"


# HTTP Errors handlers
@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally.
    app.run(host='0.0.0.0', debug=True)
