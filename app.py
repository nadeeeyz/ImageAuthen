import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Import optimized model handler
from modelhandler import OptimizedModelHandlerHF

# Initialize model handler (will load models on demand)
model_handler = None

def get_model_handler():
    """Lazy initialization of model handler"""
    global model_handler
    if model_handler is None:
        # Pilih salah satu:
        # Option 1: Use optimized ensemble (loads models one by one)
        model_handler = OptimizedModelHandlerHF()
        
        # Option 2: Use single best model (uncomment line below, comment line above)
        # from modelhandler import SingleModelHandlerHF
        # model_handler = SingleModelHandlerHF("Late SE")  # atau model terbaik lainnya
    return model_handler

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Routes
@app.route("/")
def intro():
    return render_template("intro.html")

@app.route("/acknowledgement")
def acknowledgement():
    return render_template("acknowledgement.html")

@app.route("/manualbook")
def manualbook():
    return render_template("manualbook.html")

@app.route("/classifier", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Process image in memory
                img = Image.open(file.stream)
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Create base64 for display
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_b64 = base64.b64encode(buffer.getvalue()).decode()

                # Reset buffer for model prediction
                buffer.seek(0)
                
                # Get model handler and make prediction
                handler = get_model_handler()
                predictions = handler.predict(buffer)

                return render_template(
                    "result.html",
                    original_filename=file.filename,
                    image_b64=img_b64,
                    predictions=predictions
                )
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                return redirect(request.url)
        else:
            flash("Invalid file type. Allowed: PNG, JPG, JPEG")

    return render_template("index.html")

# Health check endpoint untuk Vercel
@app.route("/health")
def health():
    return {"status": "ok"}, 200

if __name__ == "__main__":
    app.run(debug=True)
