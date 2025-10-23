from flask import Flask, render_template, request, send_file, jsonify
import cv2, numpy as np, uuid, os, time
from detectors.mtcnn_wrapper import detect_faces, blur_regions

app = Flask(__name__)
UPLOAD_DIR = "uploads"
OUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/detect")
def detect():
    blur = request.form.get("privacy") == "on"
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    in_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    file.save(in_path)

    img = cv2.imread(in_path)
    t0 = time.time()
    boxes = detect_faces(img)  # list[(x,y,w,h)]
    if blur:
        img = blur_regions(img, boxes)
    else:
        for (x, y, w, h) in boxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    latency_ms = int((time.time() - t0) * 1000)

    out_path = os.path.join(OUT_DIR, f"result_{os.path.basename(in_path)}")
    cv2.imwrite(out_path, img)

    resp = send_file(out_path, mimetype="image/jpeg")
    # Simple metrics for the UI
    resp.headers["X-Faces"] = str(len(boxes))
    resp.headers["X-LatencyMs"] = str(latency_ms)
    return resp

if __name__ == "__main__":
    app.run(debug=True)
