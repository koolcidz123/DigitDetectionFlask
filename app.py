from flask import Flask,jsonify,request

from main import get_pred

app = Flask(__name__)

@app.route("/digit",methods = ["POST"])
def pred_digit():
    image = request.files.get("digit")
    prediction = get_pred(image)
    return jsonify({"prediction":prediction}),200

app.run(debug= True)