
import streamlit as st
import pickle
import numpy as np
import tempfile
import os

from Orange.data import Domain, Table
from orangecontrib.imageanalytics.image_embedder import ImageEmbedder

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "DFU_Logistic_Model.pkcls")

st.set_page_config(page_title="DFU Image Classifier", page_icon="🩺")

st.title("Diabetic Foot Ulcer Image Classifier")
st.write("Upload an image for an educational AI classification.")

st.warning(
    "This is a student project, not a medical diagnostic tool. "
    "Its predictions cannot confirm or rule out a medical condition."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", width=350)

    if st.button("Analyse Image"):
        with st.spinner("Processing image..."):
            temp_path = None

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                with open(MODEL_PATH, "rb") as file:
                    classifier = pickle.load(file)

                with ImageEmbedder(model="squeezenet") as embedder:
                    embeddings = embedder([temp_path])

                features = np.asarray(embeddings, dtype=float)

                if features.shape[1] != len(
                    classifier.domain.attributes
                ):
                    raise ValueError(
                        "Image features do not match the saved model."
                    )

                domain = classifier.domain

                Y = np.full((1, len(domain.class_vars)), np.nan)
                metas = np.empty((1, len(domain.metas)), dtype=object)
                metas[:] = None

                data = Table(domain,features,Y,metas)

                prediction = classifier(data)[0]
                result = str(prediction)

                st.subheader("Classification Result")
                st.success("Predicted class: " + result)

            except Exception as error:
                st.error("Could not process the image: " + str(error))

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)
