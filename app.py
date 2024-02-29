from flask import Flask, request, render_template
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        file.save('uploaded_file.csv')
        return get_column_names()
    return "No file uploaded"

def get_column_names():
    df = pd.read_csv('uploaded_file.csv')
    column_names = df.columns.tolist()
    return render_template('select_columns.html', columns=column_names)

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    column1 = request.form['column1']
    column2 = request.form['column2']
    
    df = pd.read_csv('uploaded_file.csv')

     # Extract the data
    x = df[column1]
    y = df[column2]

    # Perform linear regression manually
    slope, intercept = np.polyfit(x, y, 1)
    line = slope * x + intercept

    # Plot the data and regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=50, label='Data')
    plt.plot(x, line, color='red', label='Linear Regression')
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title('Linear Regression Plot')
    plt.legend()
    
 # Save the plot as an image
    image_path = 'static/plot.png'
    plt.savefig(image_path)
    plt.close()

    # Display the image in the HTML template
    return render_template('display_graph.html', image_path=image_path)

@app.route('/calculate_y', methods=['POST'])
def calculate_y():
    user_x = float(request.form['user_x'])
    column1 = request.form['column1']
    column2 = request.form['column2']

    df = pd.read_csv('uploaded_file.csv')

    # Extract X and Y values from the DataFrame
    x_values = df[request.form['column1']].values.reshape(-1, 1)
    y_values = df[request.form['column2']].values

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x_values, y_values)

    # Predict the Y value for the user-provided X value
    user_y = model.predict([[user_x]])[0]

    image_path = 'static/plot.png'

    return render_template('display_graph.html', image_path=image_path, column1=column1, column2=column2, user_x=user_x, user_y=user_y, result=True)

if __name__ == '__main__':
    app.run(debug=True)
