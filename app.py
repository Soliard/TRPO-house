#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[2]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[5]:


@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 0)

    return render_template('index.html', prediction_text='Предполагаемая стоимость жилья составляет {} $'.format(output))


# In[7]:


@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


# In[8]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




