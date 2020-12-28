from classifier import Classifier
from codecs import open
import time
from flask import Flask, render_template, request
app = Flask(__name__)

print("Load classifier")
start_time = time.time()
classifier = Classifier()
print("Classifier is successfully loaded")
print(time.time() - start_time, "seconds")


@app.route("/", methods = ["POST", "GET"])
def index_page(text = "", prediction_message = ""):
    if request.method == "POST":
        text = request.form["text"]

        logfile = open("news_demo_logs.txt", "ab", "utf-8")

        # print(text)
        logfile.write(text)
        logfile.write("<response>")
        genres_list = classifier.predict_genres(text)
        prediction_message = ", ".join(genres_list)

        # print(prediction_message)
        logfile.write(prediction_message)
        logfile.write("<response>")
        logfile.close()

    return render_template('simple_page.html', text = text, prediction_message = prediction_message)

if __name__ == "__main__":
    # app.run(host = '0.0.0.0', port = 80, debug = True)
    app.run(host='127.0.0.1', port=80, debug=True)
    # app.run(host="localhost", port=8080, debug=True)
