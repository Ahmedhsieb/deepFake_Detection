# First install this in terminal:
# pip install fastapi
# pip install uvicorn
# pip install python-multipart
# pip install torch
# pip install opencv-python
# pip install Pillow
# pip install asyncio
# pip install nest_asyncio
# pip install torchvision
# pip install requests

from fastapi import FastAPI, UploadFile, File
import torch
import uvicorn
import cv2
import os
from PIL import Image
from torchvision import transforms
import requests

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

url = "https://huggingface.co/imtiyaz123/df_model.pt/resolve/main/df_model.pt"
# Check if models directory exists, if not create it
if not os.path.exists("models"):
    os.makedirs("models")

output_path = "models/df_model.pt"
response = requests.get(url)
with open(output_path, "wb") as f:
    f.write(response.content)

# Load model architecture and weights
model = torch.load("models/df_model.pt", map_location=device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def extract_frames(video_path, max_frames=30):
    frames = []
    cap = cv2.VideoCapture(video_path)
    count = 0
    success, frame = cap.read()
    while success and count < max_frames:
        frame_path = f"temp_frame_{count}.jpg"
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        count += 1
        success, frame = cap.read()
    cap.release()
    return frames

def cleanup_files(file_list):
    for f in file_list:
        if os.path.exists(f):
            os.remove(f)

@app.post("/predict/")
async def main(file: UploadFile = File(...)):
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        if file.filename.endswith((".jpg", ".jpeg", ".png")):
            img_tensor = preprocess_image(file_location)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                conf, pred = torch.max(probs, dim=1)
            result = {"prediction": pred.item(), "confidence": conf.item()}

        elif file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
            frames = extract_frames(file_location)
            preds = []
            confs = []
            for frame_path in frames:
                img_tensor = preprocess_image(frame_path)
                with torch.no_grad():
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, dim=1)
                preds.append(pred.item())
                confs.append(conf.item())
            avg_pred = round(sum(preds) / len(preds))
            avg_conf = sum(confs) / len(confs)
            result = {"prediction": avg_pred, "confidence": avg_conf}
            cleanup_files(frames)
        else:
            result = {"error": "Unsupported file type."}
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

    return result

import sys

if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        import nest_asyncio
        nest_asyncio.apply()
        import uvicorn
        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = uvicorn.Server(config)
        await server.serve()
    else:
        uvicorn.run(app, host="127.0.0.1", port=8000)
