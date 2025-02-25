from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/GetTestData', methods=['GET'])
def get_skin_tone():
    return jsonify({'Message': 'Working Fine!' , 'ResultCode' : 200})

if __name__ == '__main__':
    app.run(debug=True)
