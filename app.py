from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load HOUSE PRICE model
model = joblib.load("model.joblib")


@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # =============================
        # BASIC DETAILS
        # =============================

        area = float(request.form["area"])

        bedrooms = int(request.form["bedrooms"])

        bathrooms = int(request.form["bathrooms"])

        stories = int(request.form["stories"])

        parking = int(request.form["parking"])


        # =============================
        # YES / NO FEATURES
        # =============================

        mainroad = int(
            request.form.get("mainroad", 0)
        )

        guestroom = int(
            request.form.get("guestroom", 0)
        )

        basement = int(
            request.form.get("basement", 0)
        )

        hotwaterheating = int(
            request.form.get("hotwaterheating", 0)
        )

        airconditioning = int(
            request.form.get("airconditioning", 0)
        )

        prefarea = int(
            request.form.get("prefarea", 0)
        )


        # =============================
        # FURNISHING
        # =============================

        furnishingstatus = int(
            request.form["furnishingstatus"]
        )


        # =============================
        # CREATE DATAFRAME
        # =============================

        input_data = pd.DataFrame({

            "area": [area],

            "bedrooms": [bedrooms],

            "bathrooms": [bathrooms],

            "stories": [stories],

            "mainroad": [mainroad],

            "guestroom": [guestroom],

            "basement": [basement],

            "hotwaterheating": [hotwaterheating],

            "airconditioning": [airconditioning],

            "parking": [parking],

            "prefarea": [prefarea],

            "furnishingstatus": [furnishingstatus]

        })


        # =============================
        # IMPORTANT VERIFICATION
        # =============================

        print("\nInput Features:")

        print(input_data.columns.tolist())


        print("\nModel Features:")

        print(model.feature_names_in_)


        # =============================
        # PREDICTION
        # =============================

        prediction = model.predict(input_data)

        predicted_price = prediction[0]


        print("\nPredicted Price:")

        print(predicted_price)


        # =============================
        # SHOW RESULT
        # =============================

        return render_template(

            "index.html",

            prediction=predicted_price

        )


    except Exception as e:

        return render_template(

            "index.html",

            error=str(e)

        )


if __name__ == "__main__":

    app.run(host="0.0.0.0",port=3000 ,debug=True)
