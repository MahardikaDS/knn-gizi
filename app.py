import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

# Load your dataset
file_path = 'DataGizi.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Drop rows with missing values (optional, if you want to drop NaN rows)
data = data.dropna()

# Map jenis kelamin to numerical value
jk_mapping = {'Perempuan': 0, 'Laki-Laki': 1}  # Mapping 'Perempuan' to 0, 'Laki-Laki' to 1
data['JK_encoded'] = data['JK'].map(jk_mapping)

# Prepare data for training
x = data[['JK_encoded', 'Umur', 'BB', 'TB']]
y = data['Gizi']

# Perform imputation to handle NaN values
imputer = SimpleImputer(strategy='mean')
x = imputer.fit_transform(x)

# Train-test split after imputation
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# Streamlit app
st.title('Klasifikasi Gizi Balita')

# Collect user input for prediction
st.subheader('Masukkan Data Balita untuk Klasifikasi')
nama = st.text_input('Nama Balita:')
jk = st.selectbox('Jenis Kelamin:', ('Perempuan', 'Laki-Laki'))
umur = st.number_input('Umur (bulan):', min_value=0, max_value=120)
bb = st.number_input('Berat Badan (kg):', min_value=0.0, max_value=50.0, step=0.1)
tb = st.number_input('Tinggi Badan (cm):', min_value=0.0, max_value=200.0, step=0.1)

# Map jenis kelamin input to numerical value using mapping
jk_encoded = jk_mapping[jk]

# Predict when user clicks the 'Classify' button
if st.button('Klasifikasikan'):
    # Prepare input data for prediction
    input_data = {'JK_encoded': [jk_encoded], 'Umur': [umur], 'BB': [bb], 'TB': [tb]}
    input_df = pd.DataFrame(input_data)

    # Perform imputation for input data
    input_df = imputer.transform(input_df)

    # Predict using the KNN model
    prediction = knn.predict(input_df)

    # Display prediction result
    st.write(f'Hasil Klasifikasi untuk {nama}: {prediction[0]}')

# Show classification report and accuracy score
st.subheader('Insights:')
st.write("Data shape:", data.shape)
st.write("Akurasi KNN:", accuracy_score(knn.predict(x_test), y_test))
