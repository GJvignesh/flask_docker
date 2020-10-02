from flask import Flask, request
import pickle
from configparser import RawConfigParser
import logging

LOG_FILENAME = 'flask.log'

config = RawConfigParser()
config.read('ConfigFile.properties')


app = Flask(__name__)
app.config["DEBUG"] = True  # Setting the debug on so, no need to reload for code change
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO,  # Logging configuration
                    format='%(asctime)s::%(levelname)s::%(funcName)s::%(message)s')


@app.route('/ping')
def welcome():
    return "touched flask"


@app.route('/predict', methods=['GET'])
def predict():
    app.logger.info("request coming to /predict")

    """This method will predict the Quality of wine based on the Params:

    ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'] """

    try:
        pickle_in = open("rf_classifier.pkl", "rb")
        model = pickle.load(pickle_in)

        # Reading the incoming 9 params
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

        predicted = model.predict([[fixed_acidity, volatile_acidity, citric_acidity, sugar, chloride,
                                    free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])

        '''print([[fixed_acidity, volatile_acidity, citric_acidity, sugar, chloride,
                free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])'''

        return_response_code = config.get("Return", "success_response")
        return_response = "The Quality of wine is: " + str(predicted)
        app.logger.info(return_response_code)
        return return_response, return_response_code
        # print(predicted)

    except (ValueError, TypeError) as e:
        # print(e)
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
