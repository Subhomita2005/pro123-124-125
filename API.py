from flask import Flask,jsonify,request
from classifier import getprediction
app=Flask(__name__)
@app.route("/predict-digit",methods=["POST"])
def predictdata():
    image=request.files.get("digit")
    prediction=getprediction(image)
    return jsonify({
        "prediction":prediction
    }),200
if (__name__=="__main__"):
    app.run(debug=True)    