import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.tree import export_graphviz
from PIL import Image
import graphviz
import os

# Load models
fish_model_svm = pd.read_pickle('Supervised/SVM_fish.pkl')
fruit_model_svm = pd.read_pickle('Supervised/SVM_fruit.pkl')
pumpkin_model_svm = pd.read_pickle('Supervised/SVM_pumpkin.pkl')
fish_model_rfc = pd.read_pickle('Supervised/RFC_fish.pkl')
fruit_model_rfc = pd.read_pickle('Supervised/RFC_fruit.pkl')
pumpkin_model_rfc = pd.read_pickle('Supervised/RFC_pumpkin.pkl')
wine_model_kmeans = pd.read_pickle('Unsupervised/kmean_wine.pkl')

# Page title
st.title('Prediksi Machine Learning')

# Select classification category
st.write("### Pilih Kategori")
option = st.selectbox("Klasifikasi:", ("Fish", "Fruit", "Pumpkin", "Wine"))

# Select algorithm
if option == "Wine":
    st.write("### Pilih Algoritma")
    algorithm = "K-Means"
else:
    st.write("### Pilih Algoritma")
    algorithm = st.selectbox("Algoritma:", ("SVM", "Random Forest"))

st.markdown("---")

# Dictionaries for fish, fruit, and pumpkin types
fish_types = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conchonius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}

fruit_types = {0: "Grapefruit", 1: "Orange"}

pumpkin_types = {0: "\u00c7er\u00e7evelik", 1: "\u00dcrg\u00fcp Sivrisi"}

# Input form based on category
with st.form(key='prediction_form'):
    if option == "Fish":
        st.write("### \ud83d\udc1f Masukkan Data Ikan")
        weight = st.number_input('Berat Ikan (dalam gram)', min_value=0.0, format="%.2f")
        length = st.number_input('Panjang Ikan (dalam cm)', min_value=0.0, format="%.2f")
        height = st.number_input('Tinggi Ikan (dalam cm)', min_value=0.0, format="%.2f")
        
        submit = st.form_submit_button(label='Prediksi Jenis Ikan')
        
        if submit:
            input_data = np.array([weight, length, height]).reshape(1, -1)
            
            if algorithm == "SVM":
                prediction = fish_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = fish_model_rfc.predict(input_data)

            fish_result = fish_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Ikan: {fish_result}")
            st.balloons()

            # Visualize one tree from the Random Forest
            if algorithm == "Random Forest":
                st.write("### Visualisasi Pohon Keputusan (Random Forest):")
                if hasattr(fish_model_rfc, "estimators_"):
                    # Select the first tree for visualization
                    estimator = fish_model_rfc.estimators_[0]
                    dot_data = export_graphviz(
                        estimator,
                        out_file=None,
                        feature_names=["Weight", "Length", "Height"],
                        class_names=list(fish_types.values()),
                        filled=True,
                        rounded=True,
                        special_characters=True
                    )
                    st.graphviz_chart(dot_data)
                else:
                    st.warning("Model Random Forest tidak memiliki estimators_ untuk divisualisasikan.")

    elif option == "Fruit":
        st.write("### \ud83c\udf4e Masukkan Data Buah")
        diameter = st.number_input('Diameter Buah (dalam cm)', min_value=0.0, format="%.2f")
        weight = st.number_input('Berat Buah (dalam gram)', min_value=0.0, format="%.2f")
        red = st.slider('Skor Warna Buah Merah', 0, 255, 0)
        green = st.slider('Skor Warna Buah Hijau', 0, 255, 0)
        blue = st.slider('Skor Warna Buah Biru', 0, 255, 0)

        submit = st.form_submit_button(label='Prediksi Jenis Buah')

        if submit:
            input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)

            if algorithm == "SVM":
                prediction = fruit_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = fruit_model_rfc.predict(input_data)

            fruit_result = fruit_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Buah: {fruit_result}")
            st.balloons()

    elif option == "Pumpkin":
        st.write("### \ud83c\udf83 Masukkan Data Labu")
        area = st.number_input('Area (dalam cm\u00b2)', min_value=0.0, format="%.2f")
        perimeter = st.number_input('Keliling (dalam cm)', min_value=0.0, format="%.2f")
        major_axis_length = st.number_input('Panjang Sumbu Mayor (dalam cm)', min_value=0.0, format="%.2f")
        minor_axis_length = st.number_input('Panjang Sumbu Minor (dalam cm)', min_value=0.0, format="%.2f")
        convex_area = st.number_input('Area Cembung (dalam cm\u00b2)', min_value=0.0, format="%.2f")
        equiv_diameter = st.number_input('Diameter Ekivalen (dalam cm)', min_value=0.0, format="%.2f")
        eccentricity = st.number_input('Eksentrisitas', min_value=0.0, format="%.2f")
        solidity = st.number_input('Kepadatan', min_value=0.0, format="%.2f")
        extent = st.number_input('Ekstensi', min_value=0.0, format="%.2f")
        roundness = st.number_input('Kebulatan', min_value=0.0, format="%.2f")
        aspect_ratio = st.number_input('Rasio Aspek', min_value=0.0, format="%.2f")
        compactness = st.number_input('Kompak', min_value=0.0, format="%.2f")

        submit = st.form_submit_button(label='Prediksi Jenis Labu')

        if submit:
            input_data = np.array([area, perimeter, major_axis_length, minor_axis_length, convex_area, 
                                   equiv_diameter, eccentricity, solidity, extent, roundness, aspect_ratio, 
                                   compactness]).reshape(1, -1)

            if algorithm == "SVM":
                prediction = pumpkin_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = pumpkin_model_rfc.predict(input_data)

            pumpkin_result = pumpkin_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Labu: {pumpkin_result}")
            st.balloons()

    elif option == "Wine":
        st.write("### \ud83c\udf47 Masukkan Data Wine")
        alcohol = st.number_input('Kadar Alkohol', min_value=0.0, format="%.2f")
        malic_acid = st.number_input('Asam Malat', min_value=0.0, format="%.2f")
        ash = st.number_input('Abu', min_value=0.0, format="%.2f")
        ash_alcalinity = st.number_input('Kelarutan Abu', min_value=0.0, format="%.2f")
        magnesium = st.number_input('Magnesium', min_value=0.0, format="%.2f")
        total_phenols = st.number_input('Fenol Total', min_value=0.0, format="%.2f")
        flavanoids = st.number_input('Flavanoid', min_value=0.0, format="%.2f")
        nonflavanoid_phenols = st.number_input('Fenol Nonflavanoid', min_value=0.0, format="%.2f")
        proanthocyanins = st.number_input('Proantosianin', min_value=0.0, format="%.2f")
        color_intensity = st.number_input('Intensitas Warna', min_value=0.0, format="%.2f")
        hue = st.number_input('Hue', min_value=0.0, format="%.2f")
        od280 = st.number_input('OD280/OD315 dari Anggur', min_value=0.0, format="%.2f")
        proline = st.number_input('Prolin', min_value=0.0, format="%.2f")

        submit = st.form_submit_button(label='Prediksi Kategori Wine')

        if submit:
            input_data = np.array([alcohol, total_phenols]).reshape(1, -1)
            prediction = wine_model_kmeans.predict(input_data)
            st.success(f"### Kategori Wine: {prediction[0]}")
            st.balloons()

# Style
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
    }
    .stSlider>.stSliderHeader {
        color: #FF6347;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.tree import export_graphviz
import graphviz

# Load models
fish_model_svm = pd.read_pickle('Supervised/SVM_fish.pkl')
fruit_model_svm = pd.read_pickle('Supervised/SVM_fruit.pkl')
pumpkin_model_svm = pd.read_pickle('Supervised/SVM_pumpkin.pkl')
fish_model_rfc = pd.read_pickle('Supervised/RFC_fish.pkl')
fruit_model_rfc = pd.read_pickle('Supervised/RFC_fruit.pkl')
pumpkin_model_rfc = pd.read_pickle('Supervised/RFC_pumpkin.pkl')
wine_model_kmeans = pd.read_pickle('Unsupervised/kmean_wine.pkl')

# Page title
st.title('Prediksi Machine Learning')

# Select classification category
st.write("### Pilih Kategori")
option = st.selectbox("Klasifikasi:", ("Fish", "Fruit", "Pumpkin", "Wine"))

# Select algorithm
if option == "Wine":
    st.write("### Pilih Algoritma")
    algorithm = "K-Means"
else:
    st.write("### Pilih Algoritma")
    algorithm = st.selectbox("Algoritma:", ("SVM", "Random Forest"))

st.markdown("---")

# Dictionaries for fish, fruit, and pumpkin types
fish_types = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conchonius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}

fruit_types = {0: "Grapefruit", 1: "Orange"}

pumpkin_types = {0: "Çerçevelik", 1: "Ürgüp Sivrisi"}

# Input form based on category
with st.form(key='prediction_form'):
    if option == "Fish":
        st.write("### Masukkan Data Ikan")
        weight = st.number_input('Berat Ikan (dalam gram)', min_value=0.0, format="%.2f")
        length = st.number_input('Panjang Ikan (dalam cm)', min_value=0.0, format="%.2f")
        height = st.number_input('Tinggi Ikan (dalam cm)', min_value=0.0, format="%.2f")
        
        submit = st.form_submit_button(label='Prediksi Jenis Ikan')
        
        if submit:
            input_data = np.array([weight, length, height]).reshape(1, -1)
            
            if algorithm == "SVM":
                prediction = fish_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = fish_model_rfc.predict(input_data)

                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = fish_model_rfc.estimators_[0]  # Get the first tree in the forest
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Weight", "Length", "Height"],
                    class_names=list(fish_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)

            fish_result = fish_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Ikan: {fish_result}")
            st.balloons()

    elif option == "Fruit":
        st.write("### Masukkan Data Buah")
        diameter = st.number_input('Diameter Buah (dalam cm)', min_value=0.0, format="%.2f")
        weight = st.number_input('Berat Buah (dalam gram)', min_value=0.0, format="%.2f")
        red = st.slider('Skor Warna Buah Merah', 0, 255, 0)
        green = st.slider('Skor Warna Buah Hijau', 0, 255, 0)
        blue = st.slider('Skor Warna Buah Biru', 0, 255, 0)
        
        submit = st.form_submit_button(label='Prediksi Jenis Buah')
        
        if submit:
            input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)
            
            if algorithm == "SVM":
                prediction = fruit_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = fruit_model_rfc.predict(input_data)

                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = fruit_model_rfc.estimators_[0]  # Get the first tree in the forest
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Diameter", "Weight", "Red", "Green", "Blue"],
                    class_names=list(fruit_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)

            fruit_result = fruit_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Buah: {fruit_result}")
            st.balloons()

    elif option == "Pumpkin":
        st.write("### Masukkan Data Labu")
        area = st.number_input('Area (dalam cm\u00b2)', min_value=0.0, format="%.2f")
        perimeter = st.number_input('Keliling (dalam cm)', min_value=0.0, format="%.2f")
        major_axis_length = st.number_input('Panjang Sumbu Mayor (dalam cm)', min_value=0.0, format="%.2f")
        minor_axis_length = st.number_input('Panjang Sumbu Minor (dalam cm)', min_value=0.0, format="%.2f")
        convex_area = st.number_input('Area Cembung (dalam cm\u00b2)', min_value=0.0, format="%.2f")
        equiv_diameter = st.number_input('Diameter Ekivalen (dalam cm)', min_value=0.0, format="%.2f")
        eccentricity = st.number_input('Eksentrisitas', min_value=0.0, format="%.2f")
        solidity = st.number_input('Kepadatan', min_value=0.0, format="%.2f")
        extent = st.number_input('Ekstensi', min_value=0.0, format="%.2f")
        roundness = st.number_input('Kebulatan', min_value=0.0, format="%.2f")
        aspect_ratio = st.number_input('Rasio Aspek', min_value=0.0, format="%.2f")
        compactness = st.number_input('Kompak', min_value=0.0, format="%.2f")

        
        submit = st.form_submit_button(label='Prediksi Jenis Labu')
        
        if submit:
            input_data = np.array([area, perimeter, major_axis_length, minor_axis_length, convex_area, equiv_diameter, eccentricity, solidity, extent, roundness, aspect_ratio, compactness]).reshape(1, -1)
            
            if algorithm == "SVM":
                prediction = pumpkin_model_svm.predict(input_data)
            else:  # Random Forest
                prediction = pumpkin_model_rfc.predict(input_data)

                # Visualize tree
                st.write("### Visualisasi Pohon Keputusan")
                tree = pumpkin_model_rfc.estimators_[0]  # Get the first tree in the forest
                dot_data = export_graphviz(
                    tree,
                    out_file=None,
                    feature_names=["Area", "Perimeter", "Major Axis Length", "Minor Axis Length", "Convex Area", "Equiv Diameter", "Eccentricity", "Solidity", "Extent", "Roundness", "Aspect Ratio", "Compactness"],
                    class_names=list(pumpkin_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(graph.source)

            pumpkin_result = pumpkin_types.get(prediction[0], "Unknown")
            st.success(f"### Jenis Labu: {pumpkin_result}")
            st.balloons()

    else:  # Wine
        st.write("### Masukkan Data Wine")
        alcohol = st.number_input('Kadar Alkohol', min_value=0.0, format="%.2f")
        sugar = st.number_input('Kadar Gula (dalam gram)', min_value=0.0, format="%.2f")
        acidity = st.number_input('Keasaman', min_value=0.0, format="%.2f")
        
        submit = st.form_submit_button(label='Prediksi Wine')
        
        if submit:
            input_data = np.array([alcohol, sugar, acidity]).reshape(1, -1)
            prediction = wine_model_kmeans.predict(input_data)
            st.success(f"### Kategori Wine: Cluster {prediction[0]}")
            st.balloons()


