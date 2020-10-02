from flask import Flask, request
import pickle
from configparser import RawConfigParser
import logging
from flasgger import Swagger

LOG_FILENAME = 'flasgger.log'

config = RawConfigParser()
config.read('ConfigFile.properties')


app = Flask(__name__)
Swagger(app)
app.config["DEBUG"] = True  # Setting the debug on so, no need to reload for code change
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,  # Logging configuration
                    format='%(asctime)s::%(levelname)s::%(funcName)s::%(message)s')


@app.route('/flasgger/ping')
def welcome():
    """This is just ping
    ---
    responses:
        200:
            description: Service running
    """
    return "touched flasgger"


@app.route('/flasgger/predict', methods=['GET'])
def predict():

    """This method will predict the Quality of wine based on the Params:
    ---
    parameters:  
      - name: fixed_acidity
        in: query
        type: number
        required: true
      - name: volatile_acidity
        in: query
        type: number
        required: true
      - name: citric_acidity
        in: query
        type: number
        required: true
      - name: sugar
        in: query
        type: number
        required: true
      - name: free_sulfur_dioxide
        in: query
        type: number
        required: true
      - name: total_sulfur_dioxide
        in: query
        type: number
        required: true
      - name: density
        in: query
        type: number
        required: true
      - name: chloride
        in: query
        type: number
        required: true
      - name: ph
        in: query
        type: number
        required: true
      - name: sulphates
        in: query
        type: number
        required: true
      - name: alcohol
        in: query
        type: number
        required: true
    responses:
        200:
            description: The model predicted the wine Quality

        400:
            description: Bad Request
    """
    app.logger.info("request coming to /predict")

    try:
        pickle_in = open("rf_classifier.pkl", "rb")
        model = pickle.load(pickle_in)

        # Reading the incoming 11 params
        fixed_acidity = float(request.args.get("fixed_acidity"))
        volatile_acidity = float(request.args.get("volatile_acidity"))
        citric_acidity = float(request.args.get("citric_acidity"))
        sugar = float(request.args.get("sugar"))
        free_sulfur_dioxide = float(request.args.get("free_sulfur_dioxide"))
        total_sulfur_dioxide = float(request.args.get("total_sulfur_dioxide"))
        density = float(request.args.get("density"))
        chloride = float(request.args.get("chloride"))
        ph = float(request.args.get("ph"))
        sulphates = float(request.args.get("sulphates"))
        alcohol = float(request.args.get("alcohol"))

        print([[fixed_acidity, volatile_acidity, citric_acidity, sugar, chloride,
                free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])

        predicted = model.predict([[fixed_acidity, volatile_acidity, citric_acidity, sugar, chloride,
                                    free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])

        return_response_code = config.get("Return", "success_response")
        return_response = "The Quality of wine is: " + str(predicted)
        app.logger.info(return_response_code)
        return return_response,return_response_code
        # print(predicted)

    except (ValueError, TypeError) as e:
        # print(e)
        print([[fixed_acidity, volatile_acidity, citric_acidity, sugar, chloride,
                free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])
        return_response_code = config.get("Return", "bad_response")
        return_response = config.get("Return", "param_error") + '\n\n' + config.get("Return", "note")
        app.logger.error(str(e) + return_response_code)
        return return_response, return_response_code

    except FileNotFoundError as e:
        app.logger.error(str(e))
        return "Not able to locate the model dump"

    except ModuleNotFoundError as e:
        app.logger.error(str(e))
        return "Module dependency issue"


if __name__ == '__main__':
    app.run()
