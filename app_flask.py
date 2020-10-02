from flask import Flask, request,  redirect, url_for
import pickle
from configparser import RawConfigParser
import logging
from flasgger import Swagger

LOG_FILENAME = 'flask.log'

config = RawConfigParser()
config.read('ConfigFile.properties')


app = Flask(__name__)
Swagger(app)
app.config["DEBUG"] = True  # Setting the debug on so, no need to reload for code change
logging.basicConfig(level=logging.INFO,  # Logging configuration
                    format='%(asctime)s::%(levelname)s::%(funcName)s::%(message)s')


@app.route('/ping')
def welcome():
    """ping url: 200 OK
    ---
    responses:
        200:
            description: Service running
    """
    return "touched flask"


@app.route('/')
def home():
    return redirect(url_for("welcome"))


@app.route('/predict', methods=['GET'])
def predict():
    """This method will predict the Quality of wine based on the Params:

       ['fixed acidity',
       'volatile acidity',
       'citric acid',
        'residual sugar',
        'chlorides',
        'free sulfur dioxide',
        'total sulfur dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol']
    ---
    parameters:
      - name: fixed_acidity
        in: query
        type: number
      - name: volatile_acidity
        in: query
        type: number
      - name: citric_acidity
        in: query
        type: number
      - name: sugar
        in: query
        type: number
      - name: chloride
        in: query
        type: number
      - name: free_sulfur_dioxide
        in: query
        type: number
      - name: total_sulfur_dioxide
        in: query
        type: number
      - name: density
        in: query
        type: number
      - name: ph
        in: query
        type: number
      - name: sulphates
        in: query
        type: number
      - name: alcohol
        in: query
        type: number
    responses:
        200:
            description: Predicted quality of wine
        400:
            description: Accepted input values are int and float32
    """

    app.logger.info("request coming to /predict")

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

        print([[fixed_acidity, volatile_acidity, citric_acidity, sugar, chloride,
                free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol]])

        return_response_code = config.get("Return", "success_response")
        return_response = "The Quality of wine is: " + str(predicted)
        app.logger.info(return_response_code)
        return return_response, return_response_code
        print(predicted)

    except (ValueError, TypeError) as e:
        print(e)
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
    app.run(host='0.0.0.0',port=8000)
