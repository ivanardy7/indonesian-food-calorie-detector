# Food Calory Detection - Computer Vision

> Detect Indonesian food from images and calculate total calories using YOLOv12 Object Detection.

## Description

This application uses the **YOLOv12** model to detect 12 types of Indonesian food from images, count the quantity of each food item, and convert the results into **total calories**.

### Supported Foods

| No | Food | Calories | Unit |
|----|------|----------|------|
| 1 | Ayam Goreng | 260 kal | per 100 gr |
| 2 | Capcay | 67 kal | per 100 gr |
| 3 | Nasi | 129 kal | per 100 gr |
| 4 | Sayur Bayam | 36 kal | per 100 gr |
| 5 | Sayur Kangkung | 98 kal | per 100 gr |
| 6 | Sayur Sop | 22 kal | per 100 gr |
| 7 | Tahu | 80 kal | per 100 gr |
| 8 | Telur Dadar | 93 kal | per 100 gr |
| 9 | Telur Mata Sapi | 110 kal | per 1 butir |
| 10 | Telur Rebus | 78 kal | per 1 butir |
| 11 | Tempe | 225 kal | per 100 gr |
| 12 | Tumis Buncis | 65 kal | per 100 gr |

## Architecture

```
User Upload Image → YOLOv12 Detection → Count Objects → Calculate Calories → Display Results
```

## How to Run

### 1. Model Training (Google Colab)
```bash
# Open notebooks/training.ipynb in Google Colab
# Ensure Runtime > Change runtime type > GPU (T4)
# Run all cells sequentially
# Download the best_calory_model.pt file after training is complete
```

### 2. Run Streamlit (Locally)
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure best.pt is in the root folder
# Run the application
streamlit run app.py
```

## Project Structure

```
├── app.py                  # Streamlit Application
├── best.pt                 # YOLOv12 model (trained weights)
├── requirements.txt        # Python dependencies
├── packages.txt            # System packages (Streamlit Cloud)
├── notebooks/
│   └── training.ipynb      # Training notebook (Google Colab)
└── README.md
```

## Tech Stack

- **Model**: YOLOv12 (Ultralytics)
- **Dataset**: Roboflow Calory Dataset (974 images)
- **Frontend**: Streamlit
- **Language**: Python 3.10+

## Dataset

- **Source**: [Roboflow - Calory Dataset](https://universe.roboflow.com/ayu-asipq/calory/dataset/1)
- **Total Images**: 974 (Train: 721, Valid: 147, Test: 106)
- **Image Size**: 640x640 pixels
- **Format**: YOLOv12
