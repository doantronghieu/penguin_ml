import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write('This app uses 06 inputs to predict the species of penguin using'
         'a model built on the Penguin dataset. Use the form below to get'
         'started.')

password_guess = st.text_input("Password?")
if (password_guess != st.secrets['password']): st.stop()

penguin_file = st.file_uploader('Upload data')
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
unique_penguin_mapping = pickle.load(map_pickle)
rfc = pickle.load(rf_pickle)
rf_pickle.close()
map_pickle.close()
df = pd.read_csv('penguins.csv')
df = df.dropna()
output = df['species']
features = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm',
              'body_mass_g', 'sex']]
features = pd.get_dummies(features)

output, uniques = pd.factorize(output)
X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=0.8)

rfc = RandomForestClassifier(random_state=15)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
score = accuracy_score(y_pred, y_test)

st.write(f"Trained model's score: {score}")
  
with st.form('user_input'):
  island = st.selectbox('Island', options=['Biscoe', 'Dream', 'Torgerson'])
  sex = st.selectbox('Sex', options=['Male', 'Female'])
  bill_length = st.number_input('Bill Length (mm)', min_value=0)
  bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
  flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
  body_mass = st.number_input('Body Mass (g)', min_value=0)
  st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if (island == 'Biscoe'): island_biscoe = 1
elif (island == 'Dream'): island_dream = 1
elif (island == 'Torgerson'): island_torgerson = 1

sex_male, sex_female = 0, 0
if (sex == 'Male'): sex_male = 1
elif (sex == 'Female'): sex_female = 1

st.subheader("Penguin Classifier: A Machine Learning App")
prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass,
              island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])
prediction_species = unique_penguin_mapping[prediction][0]
st.write(f'Prediction: {prediction_species}')
st.write(f'We used a machine learning (Random Forest) model to '
         'predict the species, the features used in this prediction'
         ' are ranked by relative importance below.')
st.image('feature_importance.png')

st.write('Below are the histograms for each continuous variable'
         'separated by penguin species. The vertical line'
         'represents your the inputted value.')
fig, ax = plt.subplots()
ax = sns.displot(x=df['bill_length_mm'], hue=df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=df['bill_depth_mm'], hue=df['species'])
plt.axvline(bill_length)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=df['flipper_length_mm'], hue=df['species'])
plt.axvline(bill_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)
