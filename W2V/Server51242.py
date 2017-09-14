# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request
from RequestHandler import RequestHandler
import time
# server.
app = Flask(__name__)
rHandler = RequestHandler()
# filename = "smp.log"
@app.route('/smp2017_ecdt', methods=['POST'])
def returnPost():
    # print(request.get_json())
    sentencesList = request.json.get('sentencesList')
    resultsList = rHandler.getBatchResults(sentencesList)
    return jsonify({'resultsList': resultsList})

if __name__ == '__main__':
    # app.run(host='119.23.174.193', port=51243)
    app.run(host='0.0.0.0', port=51242)
