import pandas as pd
import  pickle

from flask import Flask, request,jsonify
from flask_cors import CORS

app = Flask(__name__)

#mengaktifkan CORS untuk mengizinkan semua domain
CORS(app)

#Memuat model yang sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return "<h1>Selamat Datang di API DS MODEL</h1>"

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    #gunakan try catch untuk memnampilkan gangguan atau error
    try:
        # Mendapatkan data dari request
        data = request.get_json()

        #input untuk memprediksi
        #prediksi diabetes berdasarkan faktor-faktor 
        input_data = pd.DataFrame([{
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age']
        }])

        #melakukan prediksi
        predictions = model.predict(input_data)

        probabilities = model.predict_proba(input_data)

        #Probabilitas positif dan negatif dalam bentuk persentase
        probabilitity_negative = probabilities[0][0] * 100; # probabilitas untuk kelas 0 negatif
        probability_positive = probabilities[0][1] * 100; # probabilitas untuk kelas 1 positif

        if predictions[0] == 1:
            result = f'Anda memiliki peluang menderita diabetes berdasarkan model KNN kami. Kemungkinan menderita diabetes adalah {probability_positive:.2f}%'
        else:
                result = "Hasil prediksi menunjukkan anda kemungkinan rendah diabetes"

            #Menampilkan hasil prediksi dan probabilitas dalam bentuk JSON
        return jsonify({
             'predictions':result,
             'probabilities':{
             'negative':f"{probabilitity_negative:.2f}%",
             'positive':f"{probability_positive:.2f}%"
             }
             })
    except Exception as e:
        return jsonify({'error':str(e)}),400
    

if __name__ == '__main__':
    app.run(debug=True)