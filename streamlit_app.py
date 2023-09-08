import streamlit as st
from PIL import Image
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration

# Custom CSS
st.markdown(
    """
    <style>
        .reportview-container {
            background: #F0F2F6;
        }
        .sidebar .sidebar-content {
            background: #F0F2F6;
        }
        h1 {
            color: #1F77B4;
        }
        h2 {
            color: #FF7F0E;
        }
        .stButton>button {
            background-color: #1F77B4;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize the processor and model (make sure the models are downloaded in your environment)
processor = Pix2StructProcessor.from_pretrained('google/deplot')
model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')

# Add a sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Home", "About"])

if page == "Home":
    # Streamlit application title
    st.title("Graph to Data Converter")

    # Upload image through streamlit
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    # Check if an image is uploaded
    if uploaded_image is not None:
        # Open the image with PIL (Pillow)
        image = Image.open(uploaded_image)

        # Display image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Add a button to generate structure
        if st.button('Generate Structure'):
            # Prepare the image and text for input
            inputs = processor(images=image, text="Generate underlying data table of the figure below:",
                               return_tensors="pt")

            # Generate predictions
            predictions = model.generate(**inputs, max_new_tokens=512)

            # Decode and display the predictions
            st.subheader("Generated Data:")
            decoded_text = processor.decode(predictions[0], skip_special_tokens=True)
            st.write(
                f'<div style="background-color: #f2f2f2; padding: 10px; border-radius: 5px; margin-top: 10px;">{decoded_text}</div>',
                unsafe_allow_html=True)

elif page == "About":
    st.title("About this App")
    st.write("This is a application demonstrating how to convert graphs to structured data.")
    st.write("It uses Streamlit for the web interface and Hugging Face's Transformers library for the model.")
