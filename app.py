from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import translate
from absl import flags
from utils import tokenizer

_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6

# define the app
app = Flask(__name__)
CORS(app)  # needed for cross-domain requests, allow everything by default

# logging for heroku
if 'DYNO' in os.environ:
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.INFO)

FLAGS = flags.FLAGS


# API route
@app.route('/api', methods=['POST'])
def api():
    import transformer_main

    tf.logging.set_verbosity(tf.logging.INFO)

    input_data = request.json
    app.logger.info("api_input: " + str(input_data))

    subtokenizer = tokenizer.Subtokenizer(FLAGS.vocab_file)

    # Set up estimator and params
    params = transformer_main.PARAMS_MAP[FLAGS.param_set]
    params["beam_size"] = _BEAM_SIZE
    params["alpha"] = _ALPHA
    params["extra_decode_length"] = _EXTRA_DECODE_LENGTH
    params["batch_size"] = _DECODE_BATCH_SIZE
    estimator = tf.estimator.Estimator(
        model_fn=transformer_main.model_fn, model_dir=FLAGS.model_dir,
        params=params)

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
