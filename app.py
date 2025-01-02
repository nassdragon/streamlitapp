import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_sidebar():
    st.session_state.page = "sidebar"

# Load models
fish_model_knn = pd.read_pickle('Algoritma KNN/fish.pkl')
fruit_model_knn = pd.read_pickle('Algoritma KNN/fruit.pkl')
fish_model_bayes = pd.read_pickle('Algoritma Naive-Bayes/fish_bayes.pkl')
fruit_model_bayes = pd.read_pickle('Algoritma Naive-Bayes/fruit_bayes.pkl')
fish_model_id3 = pd.read_pickle('Algoritma ID3/fish_id3.pkl')
fruit_model_id3 = pd.read_pickle('Algoritma ID3/fruit_id3.pkl')
fish_model_svm = pd.read_pickle('Supervised/SVM_fish.pkl')
fruit_model_svm = pd.read_pickle('Supervised/SVM_fruit.pkl')
pumpkin_model_svm = pd.read_pickle('Supervised/SVM_pumpkin.pkl')
fish_model_rfc = pd.read_pickle('Supervised/RFC_fish.pkl')
fruit_model_rfc = pd.read_pickle('Supervised/RFC_fruit.pkl')
pumpkin_model_rfc = pd.read_pickle('Supervised/RFC_pumpkin.pkl')
wine_model_kmeans = pd.read_pickle('Unsupervised/kmean_wine.pkl')

# Home page
if st.session_state.page == "home":
    st.title("Aplikasi Prediksi Machine Learning")
    st.write("Selamat datang di aplikasi prediksi berbasis Machine Learning!")
    st.write("Gunakan aplikasi ini untuk melakukan prediksi pada berbagai kategori data menggunakan algoritma yang berbeda.")
    if st.button("Mulai"):
        go_to_sidebar()

# Sidebar navigation
elif st.session_state.page == "sidebar":
    st.sidebar.title("Navigasi")
    prediksi_menu = st.sidebar.radio("Pilih Menu", ["Prediksi 1", "Prediksi 2"])

    if prediksi_menu == "Prediksi 1":
        st.title("Prediksi 1")
        st.write("**Prediksi Machine Learning Menggunakan Agortima KNN, Naive Bayes dan ID3.**")

        # Pilih kategori
        st.write("### Pilih Kategori")
        option = st.selectbox("Klasifikasi:", ("Fish", "Fruit"))

        # Pilih algoritma
        st.write("### Pilih Algoritma")
        algorithm = st.selectbox("Algoritma:", ("KNN", "Naive Bayes", "ID3"))

        st.markdown("---")

        # Dictionaries for fish and fruit types
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

        # Input form based on category
        with st.form(key='my_form'):
            if option == "Fish":
                st.write("### 🐟 Masukkan Data Ikan")
                weight = st.number_input('Berat Ikan (dalam gram)', min_value=0.0, format="%.2f")
                length = st.number_input('Panjang Ikan (dalam cm)', min_value=0.0, format="%.2f")
                height = st.number_input('Tinggi Ikan (dalam cm)', min_value=0.0, format="%.2f")
        
                submit = st.form_submit_button(label='Prediksi Jenis Ikan', help="Klik untuk melihat hasil prediksi")
        
            if submit:
                    input_data = np.array([weight, length, height]).reshape(1, -1)
            
                    # Memilih algoritma KNN, Naive Bayes, atau ID3
                    if algorithm == "KNN":
                        prediction = fish_model_knn.predict(input_data)
                        fish_result = prediction[0]
                    elif algorithm == "Naive Bayes":
                        prediction = fish_model_bayes.predict(input_data)
                        fish_result = fish_types.get(prediction[0], "Unknown")
                    else:  # ID3
                        prediction = fish_model_id3.predict(input_data)
                        fish_result = fish_types.get(prediction[0], "Unknown")
                    st.success(f"### Jenis Ikan: {fish_result}")

 # Visualisasi id3
            if algorithm == "ID3":
                st.write("### Visualisasi Pohon Keputusan (ID3) untuk Prediksi Ikan:")
                dot_data = export_graphviz(
                    fish_model_id3,
                    out_file=None,
                    feature_names=["Weight", "Length", "Height"],
                    class_names=list(fish_types.values()),
                    filled=True,
                    rounded=True,
                    special_characters=True
                )
                graph = graphviz.Source(dot_data)
                st.graphviz_chart(dot_data)


            elif option == "Fruit":
                st.write("### 🍎 Masukkan Data Buah")
                diameter = st.number_input('Diameter Buah (dalam cm)', min_value=0.0, format="%.2f")
                weight = st.number_input('Berat Buah (dalam gram)', min_value=0.0, format="%.2f")
                red = st.slider('Skor Warna Buah Merah', 0, 255, 0)
                green = st.slider('Skor Warna Buah Hijau', 0, 255, 0)
                blue = st.slider('Skor Warna Buah Biru', 0, 255, 0)
        
                submit = st.form_submit_button(label='Prediksi Jenis Buah', help="Klik untuk melihat hasil prediksi")
        
            if submit:
                    input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)
            
                    # Memilih algoritma KNN atau Naive Bayes
                    if algorithm == "KNN":
                        prediction = fruit_model_knn.predict(input_data)
                    else:  # Naive Bayes
                        prediction = fruit_model_bayes.predict(input_data)
            
                    # Mengubah hasil prediksi numerik ke kategori
                    fruit_result = fruit_types.get(prediction[0], "Unknown")
            
                    st.success(f"### Jenis Buah: {fruit_result}")

    elif prediksi_menu == "Prediksi 2":
        st.title("Prediksi 2")
        st.write("**Prediksi Machine Learning Menggunakan Agortima SVM, Random Forest dan K-Means.**")

        # Pilih kategori
        st.write("### Pilih Kategori")
        option = st.selectbox("Klasifikasi:", ("Fish", "Fruit", "Pumpkin", "Wine"))

        # Pilih algoritma
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
                        tree = fish_model_rfc.estimators_[0]
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
                        tree = fruit_model_rfc.estimators_[0]
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

            elif option == "Pumpkin":
                st.write("### Masukkan Data Labu")
                area = st.number_input('Area (dalam cm²)', min_value=0.0, format="%.2f")
                perimeter = st.number_input('Keliling (dalam cm)', min_value=0.0, format="%.2f")
                major_axis_length = st.number_input('Panjang Sumbu Mayor (dalam cm)', min_value=0.0, format="%.2f")
                minor_axis_length = st.number_input('Panjang Sumbu Minor (dalam cm)', min_value=0.0, format="%.2f")
                convex_area = st.number_input('Area Cembung (dalam cm²)', min_value=0.0, format="%.2f")
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
                    st.success(f"### Jenisa Labu: {pumpkin_result}")

if prediction_type == "Wine":  # Kondisi untuk Wine
    st.write("### Klasifikasi Data Wine")

    try:
        # Load model .pkl
        with open("Unsupervised/kmean_wine.pkl", "rb") as file:
            kmeans_model = pickle.load(file)

        # Simulasikan dataset
        wine_data = pd.DataFrame({
            "alcohol": np.random.uniform(10, 15, 200),
            "total_phenols": np.random.uniform(0.1, 5, 200),
        })
        wine_features = wine_data.values

        # Input untuk jumlah maksimal K
        max_k = st.slider("Pilih Maksimal K", min_value=2, max_value=20, value=10)

        # Tombol untuk menampilkan Elbow Method
        if st.button("Tampilkan Grafik Elbow Method"):
            # Hitung SSE untuk setiap nilai K
            sse = []
            for k in range(1, max_k + 1):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(wine_features)
                sse.append(kmeans.inertia_)

            # Plot grafik Elbow Method
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, max_k + 1), sse, marker='o')
            plt.xlabel("Jumlah Kluster (K)")
            plt.ylabel("Sum of Squared Errors (SSE)")
            plt.title("Elbow Method untuk Menentukan Nilai Optimal K")
            plt.grid(True)

            # Tampilkan grafik di Streamlit
            st.pyplot(plt)

    except FileNotFoundError:
        st.error("Model kmean_wine.pkl tidak ditemukan! Pastikan file ada di direktori yang sama.")
else:
    st.write("Pilih jenis prediksi lain.")

