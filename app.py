# Import Libraries
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
from collections import Counter

# Web App Tab Configuration
st.set_page_config(
    page_title="🍽️ Food Calory Detection",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { font-family: 'Inter', sans-serif; }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35, #F7C948, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #F7C948;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    
    .food-item {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 4px solid #4ECDC4;
    }
    
    .food-name { font-weight: 600; color: #fff; }
    .food-cal { color: #F7C948; font-weight: 600; }
    
    .total-box {
        background: linear-gradient(135deg, #FF6B35, #F7C948);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    
    .total-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .total-label {
        color: rgba(26,26,46,0.7);
        font-weight: 500;
    }
    
    .sidebar .sidebar-content { background: #0e1117; }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #F7C948);
        color: #1a1a2e;
        font-weight: 600;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255,107,53,0.4);
    }
    
    .info-box {
        background: rgba(78, 205, 196, 0.1);
        border: 1px solid rgba(78, 205, 196, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CLASS_NAMES = [
    'Ayam Goreng', 'Capcay', 'Nasi', 'Sayur Bayam',
    'Sayur Kangkung', 'Sayur Sop', 'Tahu', 'Telur Dadar',
    'Telur Mata Sapi', 'Telur Rebus', 'Tempe', 'Tumis Buncis',
    'Food (Other)'
]

CALORY_MAP = {
    0: 260, 1: 67, 2: 129, 3: 36, 4: 98, 5: 22,
    6: 80, 7: 93, 8: 110, 9: 78, 10: 225, 11: 65, 12: 0
}

CALORY_UNIT = {
    0: '100 gr', 1: '100 gr', 2: '100 gr', 3: '100 gr',
    4: '100 gr', 5: '100 gr', 6: '100 gr', 7: '100 gr',
    8: '1 butir', 9: '1 butir', 10: '100 gr', 11: '100 gr', 12: '-'
}

FOOD_EMOJI = {
    0: '🍗', 1: '🥬', 2: '🍚', 3: '🥬', 4: '🥬', 5: '🍲',
    6: '🧈', 7: '🍳', 8: '🍳', 9: '🥚', 10: '🫘', 11: '🥗', 12: '🍽️'
}

# Model Loading
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("❌ Model file `best.pt` not found! Please place it in the app directory.")
        st.stop()
    model = YOLO(model_path)
    return model

def detect_food(model, image, conf_threshold=0.25):
    # Convert PIL (RGB) to OpenCV (BGR) for correct color inference
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Run prediction with correct color channels
    results = model.predict(img_bgr, conf=conf_threshold, verbose=False)
    result = results[0]
    
    detections = []
    food_count = Counter()
    total_calories = 0
    
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'Unknown-{cls_id}'
            calories = CALORY_MAP.get(cls_id, 0)
            
            detections.append({
                'class_id': cls_id,
                'name': name,
                'confidence': confidence,
                'calories': calories,
                'unit': CALORY_UNIT.get(cls_id, '-'),
                'emoji': FOOD_EMOJI.get(cls_id, '🍽️'),
                'bbox': box.xyxy[0].tolist()
            })
            
            food_count[cls_id] += 1
            total_calories += calories
    
    # Process output image channels
    annotated = result.plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    return {
        'detections': detections,
        'food_count': dict(food_count),
        'total_calories': total_calories,
        'num_items': len(detections),
        'annotated_image': annotated_rgb
    }

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Minimum confidence score for detection"
    )
    
    st.markdown("---")
    
    st.markdown("## 📋 Supported Foods")
    for i, name in enumerate(CLASS_NAMES):
        if i < 12:  # Skip 'Food (Other)'
            emoji = FOOD_EMOJI[i]
            cal = CALORY_MAP[i]
            unit = CALORY_UNIT[i]
            st.markdown(f"{emoji} **{name}** — {cal} cal/{unit}")
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#666; font-size:0.8rem;'>"
        "Computer Vision - Food Calory Detection</p>",
        unsafe_allow_html=True
    )

# Web App Interface
st.markdown('<h1 class="hero-title">🍽️ Food Calory Detection</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">'
    'Upload food images to detect food types and calculate total calories'
    '</p>',
    unsafe_allow_html=True
)

# Load model weight
model = load_model()

# Image file uploader
uploaded_file = st.file_uploader(
    "📸 Upload food image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded_file is not None:
    # Read uploaded file
    image = Image.open(uploaded_file).convert("RGB")
    
    # Start detection pipeline
    with st.spinner("🔍 Detecting food items..."):
        result = detect_food(model, image, conf_threshold)
    
    # Layout grid
    col_img, col_result = st.columns([3, 2])
    
    with col_img:
        st.markdown("### 📸 Detection Output")
        st.image(result['annotated_image'], width='stretch')
    
    with col_result:
        st.markdown("### 📊 Calorie Analysis")
        
        # Stat cards
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{result["num_items"]}</div>'
                f'<div class="metric-label">Food Detected</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{len(result["food_count"])}</div>'
                f'<div class="metric-label">Food Types</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("")
        
        # Render list of detected items
        if result['detections']:
            st.markdown("#### 🍽️ Food Details")
            
            for cls_id, count in sorted(result['food_count'].items()):
                name = CLASS_NAMES[cls_id]
                emoji = FOOD_EMOJI.get(cls_id, '🍽️')
                cal_per_item = CALORY_MAP.get(cls_id, 0)
                total_cal = cal_per_item * count
                unit = CALORY_UNIT.get(cls_id, '-')
                
                st.markdown(
                    f'<div class="food-item">'
                    f'<span class="food-name">{emoji} {name} x{count}</span>'
                    f'<span class="food-cal">{total_cal} kcal</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Show sum
            st.markdown(
                f'<div class="total-box">'
                f'<div class="total-label">🔥 TOTAL CALORIES</div>'
                f'<div class="total-value">{result["total_calories"]} kcal</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("🔍 No food detected. Try lowering the confidence threshold.")
    
    # Detailed log view
    if result['detections']:
        st.markdown("---")
        st.markdown("### 📋 Detection Details")
        
        import pandas as pd
        
        table_data = []
        for i, det in enumerate(result['detections'], 1):
            table_data.append({
                'No': i,
                'Food': f"{det['emoji']} {det['name']}",
                'Confidence': f"{det['confidence']:.1%}",
                'Calories': f"{det['calories']} kcal",
                'Unit': det['unit']
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, width='stretch', hide_index=True)

else:
    # Instruction panels
    st.markdown("")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">📸</div>'
            '<div class="metric-label">1. Upload food image</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with c2:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">🔍</div>'
            '<div class="metric-label">2. Auto-detection</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with c3:
        st.markdown(
            '<div class="metric-card">'
            '<div class="metric-value">🔥</div>'
            '<div class="metric-label">3. Get calorie estimation</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    st.markdown("")
    st.markdown(
        '<div class="info-box">'
        '<strong>ℹ️ How to Use:</strong><br>'
        '• Upload an image of Indonesian food (JPG/PNG)<br>'
        '• The system will automatically detect the food and calculate calories<br>'
        '• Adjust confidence threshold in the sidebar to fine-tune detection accuracy<br>'
        '• Supports 12 types of popular Indonesian food'
        '</div>',
        unsafe_allow_html=True
    )
