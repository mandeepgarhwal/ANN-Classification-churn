from datetime import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model

# data = pd.read_csv('Churn_Modelling.csv')
# data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
# label_encoder_gender = LabelEncoder()
# data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
# one_hot_encoder_geography = OneHotEncoder()
# geography_encoded = one_hot_encoder_geography.fit_transform(data[['Geography']])
# # print(one_hot_encoder_geography.get_feature_names_out(['Geography']))
# # print(data.head())
# geography_encoded_df = pd.DataFrame(geography_encoded.toarray(), columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
# # print(geography_encoded_df.head())
# data = pd.concat([data, geography_encoded_df], axis=1)
# data = data.drop(['Geography'], axis=1)

# with open('label_encoder_gender.pkl', 'wb') as f:
#     pickle.dump(label_encoder_gender, f)

# with open('one_hot_encoder_geography.pkl', 'wb') as f:
#     pickle.dump(one_hot_encoder_geography, f)

# x = data.drop(['Exited'], axis=1)
# y = data['Exited']

# x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)

# model = Sequential([
#     Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
#     Dense(32, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])
# opt = tf.keras.optimizers.Adam(learning_rate = 0.01)
# # print(model.summary())
# model.compile(optimizer = opt, loss = "binary_crossentropy", metrics = ["accuracy"])

# log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# model.fit(x_train, y_train, epochs=100, validation_data=(x_test,y_test), callbacks=[early_stopping, tensorboard_callback])

# model.save('churn_model.h5')

model1 = load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('one_hot_encoder_geography.pkl', 'rb') as f:
    one_hot_encoder_geography = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
label_encoded_data = one_hot_encoder_geography.transform([[input_data['Geography']]]).toarray()
data_df = pd.DataFrame(label_encoded_data, columns=one_hot_encoder_geography.get_feature_names_out(['Geography']))
# print(data_df)

input_data = pd.DataFrame([input_data])
input_data = input_data.drop(['Geography'], axis=1)
input_data = pd.concat([input_data.reset_index(drop=True), data_df], axis=1)
# print(input_data)
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])
input_data_scaled = scaler.transform(input_data)

prediction =model1.predict(input_data_scaled)
print("Prediction:", prediction)