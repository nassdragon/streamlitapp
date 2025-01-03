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
    st.title("Machine Learning Prediction Application")
    st.write("Welcome to the Machine Learning based prediction application!")
    st.write("Use this application to make predictions on various categories of data using different algorithms.")
    if st.button("Start"):
        go_to_sidebar()

# Sidebar navigation
elif st.session_state.page == "sidebar":
    st.sidebar.title("Navigation")
    prediksi_menu = st.sidebar.radio("Choose Menu", ["Prediction 1", "Prediction 2"])

    if prediksi_menu == "Prediction 1":
        st.title("Prediction 1")
        st.write("**Machine Learning Prediction Using KNN, Naive Bayes, and ID3 Algorithms.**")

        # Pilih kategori
        st.write("### Choose Category")
        option = st.selectbox("Clasification:", ("Fish", "Fruit"))

        # Pilih algoritma
        st.write("### Choose Algorithm")
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
                st.write("### 🐟 Input Fish Data")
                weight = st.number_input('Fish Weight (gram)', min_value=0.0, format="%.2f")
                length = st.number_input('Fish Length (cm)', min_value=0.0, format="%.2f")
                height = st.number_input('Fish Height (cm)', min_value=0.0, format="%.2f")
            
                submit = st.form_submit_button(label='Fish Type Prediction', help="Click to see prediction results")
            
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
                    st.success(f"### Fish Type: {fish_result}")

                    # Visualisasi pohon keputusan untuk ID3
                    if algorithm == "ID3":
                        st.write("### Decision Tree Visualization (ID3) for Fish Prediction:")
                        dot_data = export_graphviz(
                            fish_model_id3,
                            out_file=None,
                            feature_names=["Weight", "Length", "Height"],
                            class_names=list(fish_types.values()),
                            filled=True,
                            rounded=True,
                            special_characters=True
                        )
                        st.graphviz_chart(dot_data)

            elif option == "Fruit":
                st.write("### 🍎 Input FRuit Data")
                diameter = st.number_input('Fruit Diameter(cm)', min_value=0.0, format="%.2f")
                weight = st.number_input('Fruit Weight(gram)', min_value=0.0, format="%.2f")
                red = st.slider('Red Fruit Color Score', 0, 255, 0)
                green = st.slider('Green Fruit Color Score', 0, 255, 0)
                blue = st.slider('Blue Fruit Color Score', 0, 255, 0)
            
                submit = st.form_submit_button(label='Fruit Type Prediction', help="Click to see prediction results")
            
                if submit:
                    input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)
                
                    # Memilih algoritma KNN atau Naive Bayes
                    if algorithm == "KNN":
                        prediction = fruit_model_knn.predict(input_data)
                    elif algorithm == "Naive Bayes":  # Naive Bayes
                        prediction = fruit_model_bayes.predict(input_data)
                    else:  # ID3
                        prediction = fruit_model_id3.predict(input_data)
                
                    # Mengubah hasil prediksi numerik ke kategori
                    fruit_result = fruit_types.get(prediction[0], "Unknown")
                
                    st.success(f"### Fruit Type: {fruit_result}")

                    # Visualisasi pohon keputusan untuk ID3
                    if algorithm == "ID3":
                        st.write("### Decision Tree Visualization (ID3) for Fruit Prediction:")
                        dot_data = export_graphviz(
                            fruit_model_id3,
                            out_file=None,
                            feature_names=["Diameter", "Weight", "Red", "Green", "Blue"],
                            class_names=list(fruit_types.values()),
                            filled=True,
                            rounded=True,
                            special_characters=True
                        )
                        st.graphviz_chart(dot_data)


    elif prediksi_menu == "Prediction 2":
        st.title("Prediction 2")
        st.write("**Machine Learning Prediction Using Agortima SVM, Random Forest and K-Means.**")

        # Pilih kategori
        st.write("### Choose Category")
        option = st.selectbox("Clasification:", ("Fish", "Fruit", "Pumpkin", "Wine"))

        # Pilih algoritma
        if option == "Wine":
            st.write("### Choose Algorithm")
            algorithm = "K-Means"
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
                max_k = st.slider("Select Maximum K", min_value=2, max_value=20, value=10)

                # Tombol untuk menampilkan Elbow Method
                if st.button("Show Elbow Method Graph"):
                    # Hitung SSE untuk setiap nilai K
                    sse = []
                    for k in range(1, max_k + 1):
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(wine_features)
                        sse.append(kmeans.inertia_)

                    # Plot grafik Elbow Method
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, max_k + 1), sse, marker='o')
                    plt.xlabel("Number of Clusters (K)")
                    plt.ylabel("Sum of Squared Errors (SSE)")
                    plt.title("Elbow Method for Determining the Optimal Value of K")
                    plt.grid(True)

                    # Tampilkan grafik di Streamlit
                    st.pyplot(plt)

            except FileNotFoundError:
                st.error("Model kmean_wine.pkl not found! Make sure the files are in the same directory")
        else:
            st.write("### Choose Algorithm")
            algorithm = st.selectbox("Algorithm:", ("SVM", "Random Forest"))

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
                st.write("### Input Fish Data")
                weight = st.number_input('Fish Weight (gram)', min_value=0.0, format="%.2f")
                length = st.number_input('Fish Length (cm)', min_value=0.0, format="%.2f")
                height = st.number_input('Fish Height (cm)', min_value=0.0, format="%.2f")

                submit = st.form_submit_button(label='Fish Type Prediction')

                if submit:
                    input_data = np.array([weight, length, height]).reshape(1, -1)

                    if algorithm == "SVM":
                        prediction = fish_model_svm.predict(input_data)
                    else:  # Random Forest
                        prediction = fish_model_rfc.predict(input_data)

                        # Visualize tree
                        st.write("### Decision Tree Visualization")
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
                    st.success(f"### Fish Type: {fish_result}")

            elif option == "Fruit":
                st.write("### Input Fruit Data")
                diameter = st.number_input('Fruit Diameter (dalam cm)', min_value=0.0, format="%.2f")
                weight = st.number_input('Fruit Weight (dalam gram)', min_value=0.0, format="%.2f")
                red = st.slider('Red Fruit Color Score', 0, 255, 0)
                green = st.slider('Green Fruit Color Score', 0, 255, 0)
                blue = st.slider('Blue Fruit Color Score', 0, 255, 0)

                submit = st.form_submit_button(label='Fruit Type Prediction')

                if submit:
                    input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)

                    if algorithm == "SVM":
                        prediction = fruit_model_svm.predict(input_data)
                    else:  # Random Forest
                        prediction = fruit_model_rfc.predict(input_data)

                        # Visualize tree
                        st.write("### Decision Tree Visualization")
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
                    st.success(f"### Fruit Type: {fruit_result}")

            elif option == "Pumpkin":
                st.write("### Input Pumpkin Data")
                area = st.number_input('Area (dalam cm²)', min_value=0.0, format="%.2f")
                perimeter = st.number_input('Perimeter (dalam cm)', min_value=0.0, format="%.2f")
                major_axis_length = st.number_input('Major Axis Length (dalam cm)', min_value=0.0, format="%.2f")
                minor_axis_length = st.number_input('Minor Axis Legth (dalam cm)', min_value=0.0, format="%.2f")
                convex_area = st.number_input('Convex Area (dalam cm²)', min_value=0.0, format="%.2f")
                equiv_diameter = st.number_input('Equiv Diameter (dalam cm)', min_value=0.0, format="%.2f")
                eccentricity = st.number_input('Eccentrity', min_value=0.0, format="%.2f")
                solidity = st.number_input('Solidity', min_value=0.0, format="%.2f")
                extent = st.number_input('Extent', min_value=0.0, format="%.2f")
                roundness = st.number_input('Roundness', min_value=0.0, format="%.2f")
                aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, format="%.2f")
                compactness = st.number_input('Compactness', min_value=0.0, format="%.2f")

                submit = st.form_submit_button(label='Pumpkin Type Prediction')

                if submit:
                    input_data = np.array([area, perimeter, major_axis_length, minor_axis_length, convex_area, equiv_diameter, eccentricity, solidity, extent, roundness, aspect_ratio, compactness]).reshape(1, -1)
                    if algorithm == "SVM":
                        prediction = pumpkin_model_svm.predict(input_data)
                    else:  # Random Forest
                        prediction = pumpkin_model_rfc.predict(input_data)

                # Visualize tree
                        st.write("### Decision Tree Visualization")
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
                    st.success(f"### Pumpkin Type: {pumpkin_result}")


# CSS untuk footer dengan background hitam dan animasi berjalan
footer_css = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #000; /* Background hitam */
        color: #fff; /* Warna teks putih */
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        overflow: hidden; /* Untuk menyembunyikan teks di luar area */
    }

    .footer-text {
        display: inline-block;
        white-space: nowrap; /* Agar teks tidak membungkus */
        animation: scroll-left 10s linear infinite; /* Animasi berjalan */
    }

    @keyframes scroll-left {
        0% {
            transform: translateX(100%);
        }
        100% {
            transform: translateX(-100%);
        }
    }
    </style>
"""

# HTML untuk konten footer dengan animasi teks
footer_html = """
    <div class="footer">
        <div class="footer-text">
           This application was created and deployed by Muhammad Anas Ma'ruf
        </div>
    </div>
"""

# Tambahkan CSS dan HTML ke halaman
st.markdown(footer_css, unsafe_allow_html=True)
st.markdown(footer_html, unsafe_allow_html=True)






