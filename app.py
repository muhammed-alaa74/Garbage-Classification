"""
Garbage Classification — Next-Gen AI Streamlit App
===================================================
An advanced, production-grade waste classification web application powered by 
deep learning (MobileNetV2 / EfficientNetB0), featuring sleek glassmorphism UI,
live confidence analytics, interactive recycling stats, and localized Arabic disposal advice.

Run locally:
    streamlit run app.py
"""

import os
import time
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

# --------------------------------------------------------------------------
# Page Configuration & Metadata
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="MLJ | النظام الذكي لإدارة وتصنيف المخلفات",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------
# Advanced UI/UX Styling (Glassmorphism & Cyber-Eco Theme)
# --------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', 'Cairo', sans-serif;
}

/* Dynamic Animated Background */
.stApp {
    background: radial-gradient(circle at 50% 0%, #0f172a, #090d16, #020617);
    background-size: 300% 300%;
    animation: backgroundShift 20s ease infinite;
}

@keyframes backgroundShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Container Spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 5rem;
    max-width: 800px;
}

/* Headings with Gradients */
h1, h2, h3 {
    font-family: 'Cairo', 'Poppins', sans-serif;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, #38bdf8 0%, #34d399 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0px;
    letter-spacing: -0.5px;
}

.sub-header {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Sidebar Customization */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090d16 0%, #0f172a 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

section[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

/* Glassmorphism Main Prediction Card */
.prediction-box {
    background: rgba(30, 41, 59, 0.65);
    backdrop-filter: blur(25px);
    -webkit-backdrop-filter: blur(25px);
    border: 1px solid rgba(56, 189, 248, 0.25);
    padding: 30px;
    border-radius: 28px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    text-align: center;
    margin: 20px 0;
    animation: cardEntrance 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes cardEntrance {
    from { opacity: 0; transform: translateY(20px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

/* Info Cards for Top 3 Predictions */
.info-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    padding: 14px 18px;
    border-radius: 16px;
    margin-bottom: 8px;
    transition: all 0.3s ease;
}

.info-card:hover {
    background: rgba(255, 255, 255, 0.07);
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateX(4px);
}

/* Arabic Eco-Tip Box */
.eco-tip-container {
    background: linear-gradient(135deg, rgba(13, 148, 136, 0.15), rgba(30, 41, 59, 0.8));
    border-right: 5px solid #34d399;
    border-left: 1px solid rgba(255, 255, 255, 0.1);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    margin-top: 25px;
    direction: rtl;
}

/* Custom Progress Bars */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #34d399, #38bdf8);
    border-radius: 12px;
}

/* File Uploader Custom Styling */
.stFileUploader {
    background: rgba(255, 255, 255, 0.02);
    border: 2px dashed rgba(56, 189, 248, 0.4);
    border-radius: 24px;
    padding: 25px;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: #34d399;
    background: rgba(255, 255, 255, 0.05);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0284c7, #0d9488);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.6rem;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(13, 148, 136, 0.35);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(56, 189, 248, 0.5);
}

/* Stats Pill */
.stats-badge {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    color: white;
    padding: 5px 14px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Constants & Data Dictionaries
# --------------------------------------------------------------------------
IMG_SIZE = (224, 224)

CLASS_LABELS = [
    "battery", "biological", "cardboard", "clothes", "glass",
    "metal", "paper", "plastic", "shoes", "trash",
]

CLASS_META = {
    "battery":    {"name_en": "Battery",    "name_ar": "بطاريات", "color": "#facc15"},
    "biological": {"name_en": "Biological", "name_ar": "مخلفات عضوية", "color": "#34d399"},
    "cardboard":  {"name_en": "Cardboard",  "name_ar": "كرتون","color": "#fb923c"},
    "clothes":    {"name_en": "Clothes",    "name_ar": "ملابس", "color": "#f472b6"},
    "glass":      {"name_en": "Glass",      "name_ar": "زجاج", "color": "#38bdf8"},
    "metal":      {"name_en": "Metal",      "name_ar": "معادن", "color": "#94a3b8"},
    "paper":      {"name_en": "Paper",      "name_ar": "ورق", "color": "#e2e8f0"},
    "plastic":    {"name_en": "Plastic",    "name_ar": "بلاستيك", "color": "#818cf8"},
    "shoes":      {"name_en": "Shoes",      "name_ar": "أحذية","color": "#a78bfa"},
    "trash":      {"name_en": "General Trash", "name_ar": "نفايات عامة", "color": "#64748b"},
}

ARABIC_TIPS = {
    "battery": (
        "**مخلفات خطرة:** البطاريات تحتوي على معادن ثقيلة كالرصاص والكادميوم التي تلوث التربة والمياه الجوفية. "
        "لا ترمها أبداً في القمامة العادية. توجه بها إلى أقرب نقطة تجميع إلكترونيات أو محل صيانة موبايلات للتخلص الآمن منها."
    ),
    "biological": (
        "**إعادة تدوير عضوي:** هذه مخلفات عضوية ممتازة لصنع سماد الكومبوست المنزلي لغذّاء النباتات والحدائق. "
        "إن لم يكن متاحاً، ضعها في كيس منفصل عن البلاستيك والورق لتسهيل عمليات الفرز البيئي."
    ),
    "cardboard": (
        "**جاهز للتدوير:** قم بفرد الصناديق الكرتونية، إزالة الأشرطة اللاصقة والدباسات، وتأكد من حفظها جافة تماماً. "
        "الكرتون الجاف يُعاد تدويره بسهولة بالكامل أو يمكن استخدامه في التغليف اليدوي."
    ),
    "clothes": (
        "**استدامة الموضة:** إذا كانت الملابس بحالة جيدة، تبرع بها لمن يحتاجها عبر الجمعيات الخيرية. "
        "أما التالفة منها فتوجّه لنقاط تجميع الأنسجة لتحويلها إلى ألياف صناعية جديدة بدلاً من تراكمها في المقالب."
    ),
    "glass": (
        "**تدوير لا نهائي:** الزجاج قابل للتدوير بنسبة 100% دون أن يفقد جودته أبداً. "
        "اغسل الزجاجة جيداً، انزع الغطاء (لأنه يصنع من مادة مختلفة)، وضعها بحذر في حاوية الزجاج المخصصة."
    ),
    "metal": (
        "**وفر طاقة هائلة:** تدوير العبوات المعدنية يوفر طاقة تفوق تصنيعها من الصفر بمراحل. "
        "اغسل العبوة جيداً من أي سوائل أو بقايا طعام وضعها مع مفرزات المعادن أو محلات الخردة."
    ),
    "paper": (
        "**الورق النظيف:** الورق قابل للتدوير لعدّة مرات ولكن بشرط أن يكون نظيفاً وخالياً من بقع الزيوت والطعام. "
        "تجنب رمي الورق اللامع أو المطلي مع الورق العادي لأنه يسبب صعوبات في المصانع."
    ),
    "plastic": (
        "**تحدي البيئة:** البلاستيك يستغرق قروناً ليتحلل. تأكد من نظافة العبوة ومعرفة رمز إعادة التدوير الموجود أسفلها (1-7). "
        "حاول دائماً تقليل استخدام العبوات ذات الاستخدام الواحد."
    ),
    "shoes": (
        "**منح فرصة ثانية:** الأحذية الصالحة للاستخدام تسعد الكثيرين عبر التبرع بها. "
        "الأحذية التالفة توجد لها برامج تجميع متخصصة تحولها إلى مطاط يُستخدم في أرضيات ملاعب الرياضة."
    ),
    "trash": (
        "**نفايات غير مقننة:** تصنيف يضم المواد التي يتعذر تدويرها حالياً (مثل الأكياس المتعددة الطبقات والأقمشة المختلطة). "
        "قلل استهلاكها قدر الإمكان وافصلها تماماً عن المواد القابلة لإعادة التدوير."
    ),
}

# --------------------------------------------------------------------------
# Model Handling
# --------------------------------------------------------------------------
MODEL_OPTIONS = {
    "MobileNetV2 (خفيف وسريع)": "models/mobilenetv2_garbage_classifier.keras",
    "EfficientNetB0 (عالي الدقة)": "models/efficientnetb0_garbage_classifier.keras",
}

@st.cache_resource(show_spinner=False)
def load_ai_model(model_path: str):
    return tf.keras.models.load_model(model_path)

def preprocess_image(img: Image.Image, model_choice: str) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    if "EfficientNet" in model_choice:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        arr = preprocess_input(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

# --------------------------------------------------------------------------
# Sidebar Dashboard
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### لوحة التحكم")
    model_choice = st.selectbox("اختر نموذج الذكاء الاصطناعي", list(MODEL_OPTIONS.keys()))
    model_path = MODEL_OPTIONS[model_choice]
    
    st.markdown("---")
    st.markdown("### الفئات المدعومة")
    for k, v in CLASS_META.items():
        st.markdown(f"<span style='font-size: 0.9rem;'><b>{v['name_ar']}</b> ({v['name_en']})</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("MLJ — مدعوم بنماذج التعلم العميق ونقل المعرفة.")

# --------------------------------------------------------------------------
# Main User Interface
# --------------------------------------------------------------------------
st.markdown('<h1 class="main-title">♻️ MLJ</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">نظام ذكي متطور لتصنيف المخلفات وتوجيهك للطريقة المثلى لإعادة تدويرها والحفاظ على البيئة.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "اسحب وأفلت صورة العنصر هنا، أو اضغط للاختيار (JPG, PNG, WEBP)", 
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة التي تم رفعها", use_container_width=True)
    
    with col2:
        if not os.path.exists(model_path):
            st.error(f"ملف النموذج غير موجود في المسار: `{model_path}`. الرجاء التأكد من وضعه داخل مجلد models/")
        else:
            with st.spinner("🔄"):
                time.sleep(0.4) # محاكاة تفاعلية سلسة
                model = load_ai_model(model_path)
                x = preprocess_image(image, model_choice)
                preds = model.predict(x, verbose=0)[0]

            top_idx = np.argsort(preds)[::-1]
            predicted_class = CLASS_LABELS[top_idx[0]]
            confidence = float(preds[top_idx[0]]) * 100
            meta = CLASS_META[predicted_class]

            # Main Result Display Box
            st.markdown(f"""
            <div class="prediction-box">
                <span style="font-size: 55px; display: block; margin-bottom: 5px;"></span>
                <h2 style="color: {meta['color']}; margin: 0; font-size: 1.8rem; font-weight: 800;">{meta['name_ar']}</h2>
                <p style="color: #94a3b8; margin: 2px 0 12px 0; font-size: 0.95rem;">{meta['name_en']}</p>
                <div style="display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.3); padding: 6px 16px; border-radius: 30px; color: #38bdf8; font-weight: 700; font-size: 0.95rem;">
                    مستوى الثقة: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Top 3 Predictions Section
    st.markdown("### احتمالات التصنيف الثلاثة الأولى")
    for i in top_idx[:3]:
        cls = CLASS_LABELS[i]
        c_meta = CLASS_META[cls]
        conf_val = float(preds[i]) * 100
        
        st.markdown(
            f"""
            <div class="info-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight: 600; color: #f1f5f9; font-size: 1rem;">
                        {c_meta["name_ar"]} <span style="color: #94a3b8; font-size: 0.85rem;">({c_meta["name_en"]})</span>
                    </span>
                    <span class="stats-badge">
                        {conf_val:.1f}%
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(float(preds[i]))

    # Smart Eco-Tip Section
    st.markdown("### الإرشادات البيئية والتخلص الآمن")
    st.markdown(
        f"""
        <div class="eco-tip-container">
            <div style="font-size: 1.08rem; line-height: 1.9; color: #f8fafc;">
                {ARABIC_TIPS[predicted_class]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.info("يرجى رفع صورة عنصر مخلفات لنظام الذكي ليبدأ التعرف عليها وفحصها فوراً.")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>MLJ — نحو بيئة مستدامة ومجتمع أخضر ذكي </p>", 
    unsafe_allow_html=True
)