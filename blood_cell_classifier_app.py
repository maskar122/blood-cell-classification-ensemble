import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# تحميل الموديلات tflite
vgg_interpreter = tf.lite.Interpreter(model_path="C:/Users/LAP-STORE/Desktop/Amit/computer vesion/VGG16_model.tflite")
resnet_interpreter = tf.lite.Interpreter(model_path="C:/Users/LAP-STORE/Desktop/Amit/computer vesion/ResNet_model.tflite")
inception_interpreter = tf.lite.Interpreter(model_path="C:/Users/LAP-STORE/Desktop/Amit/computer vesion/inception_model.tflite")

vgg_interpreter.allocate_tensors()
resnet_interpreter.allocate_tensors()
inception_interpreter.allocate_tensors()

CLASS_NAMES = ['Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
    return image_array

def predict_probs_tflite(image_array, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    probs = output[0]
    return probs

def ensemble_predict_detailed(image_array):
    vgg_probs = predict_probs_tflite(image_array, vgg_interpreter)
    resnet_probs = predict_probs_tflite(image_array, resnet_interpreter)
    inception_probs = predict_probs_tflite(image_array, inception_interpreter)

    # احسب كل نتيجة
    vgg_idx = np.argmax(vgg_probs)
    resnet_idx = np.argmax(resnet_probs)
    inception_idx = np.argmax(inception_probs)

    # جمع النتائج في dict
    results = {
        "VGG16": (CLASS_NAMES[vgg_idx], vgg_probs[vgg_idx]),
        "Inception": (CLASS_NAMES[inception_idx], inception_probs[inception_idx]),
        "ResNet": (CLASS_NAMES[resnet_idx], resnet_probs[resnet_idx]),
    }

    # نحدد أفضل موديل حسب أعلى confidence
    best_model = max(results, key=lambda k: results[k][1])
    best_class, best_conf = results[best_model]

    return results, best_model, best_class, best_conf

def set_background():
    page_bg_img = '''
    <style>
    .stApp {
      background-image: url("https://images.unsplash.com/photo-1581090700227-798a54357d6c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1470&q=80");
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      background-attachment: fixed;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    set_background()
    st.title("Blood Cell Classification Ensemble Model")
    st.write("Upload a blood cell image for analysis using the model.Ensemble")

    uploaded_file = st.file_uploader(" choose image (JPEG, PNG)", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=" Uploaded image", use_column_width=True)

        image_array = preprocess_image(image, target_size=(224, 224))
        results, best_model, best_class, best_conf = ensemble_predict_detailed(image_array)

        # عرض نتائج كل موديل
        for model_name, (pred_class, conf) in results.items():
            st.markdown(f"**{model_name} -> {pred_class} (Confidence: {conf:.2f})**")

        st.markdown("---")
        # عرض النتيجة النهائية
        st.markdown("✅ **Final Result:**")
        st.markdown(f"**Best Model:** {best_model}")
        st.markdown(f"**Predicted Class:** {best_class} (Confidence: {best_conf:.2f})")

if __name__ == "__main__":
    main()
