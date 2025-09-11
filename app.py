# # from flask import Flask

# # app = Flask(__name__)

# # @app.route("/")
# # def home():
# #     return "Hello, Flask!"

# # if __name__ == "__main__":
# #     app.run(debug=True)

# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# from modelhandler import ModelHandler

# UPLOAD_FOLDER = "static/uploads"
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# app = Flask(__name__)
# app.secret_key = "supersecretkey"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load ensemble models
# model_handler = ModelHandler(models_dir="models")

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             flash("No file part")
#             return redirect(request.url)

#         file = request.files["file"]

#         if file.filename == "":
#             flash("No selected file")
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
#             file.save(filepath)

#             predictions = model_handler.predict(filepath)

#             return render_template("result.html",
#                                    original_filename=filename,
#                                    image_path=filepath,
#                                    predictions=predictions)
#         else:
#             flash("Invalid file type. Allowed: PNG, JPG, JPEG")

#     return render_template("index.html")

# if __name__ == "__main__":
#     app.run(debug=True)


# import os
# from flask import Flask, render_template, request, redirect, url_for, flash
# from werkzeug.utils import secure_filename
# # from modelhandler import ModelHandler
# import base64
# from io import BytesIO
# from PIL import Image

# from modelhandler import ModelHandlerHF

# model_handler = ModelHandlerHF()


# UPLOAD_FOLDER = "static/uploads"
# ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# app = Flask(__name__)   # <-- ini WAJIB ada sebelum pakai @app.route
# app.secret_key = "supersecretkey"
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load ensemble models
# # model_handler = ModelHandler(models_dir="models")

# def allowed_file(filename):
#     return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# # ---------------- ROUTES ----------------
# @app.route("/")
# def intro():
#     return render_template("intro.html")
# @app.route("/acknowledgement")
# def acknowledgement():
#     return render_template("acknowledgement.html")
# @app.route("/manualbook")
# def manualbook():
#     return render_template("manualbook.html")
# @app.route("/classifier", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         if "file" not in request.files:
#             flash("No file part")
#             return redirect(request.url)

#         file = request.files["file"]

#         if file.filename == "":
#             flash("No selected file")
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             # Membaca file ke memory
#             img = Image.open(file.stream)
#             buffer = BytesIO()
#             img.save(buffer, format="PNG")
#             img_b64 = base64.b64encode(buffer.getvalue()).decode()

#             # Jika model handler membutuhkan file path
#             buffer.seek(0)
#             predictions = model_handler.predict(buffer)

#             # Kirim base64 ke template
#             return render_template(
#                 "result.html",
#                 original_filename=file.filename,
#                 image_b64=img_b64,
#                 predictions=predictions
#             )
#         else:
#             flash("Invalid file type. Allowed: PNG, JPG, JPEG")

#     return render_template("index.html")


# if __name__ == "__main__":
#     app.run(debug=True)



import os
from flask import Flask, render_template, request, redirect, flash
import base64
from io import BytesIO
from PIL import Image

from modelhandler import ModelHandlerHF

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model Handler
model_handler = ModelHandlerHF()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- ROUTES ----------------
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
            # Membaca file ke memory
            img = Image.open(file.stream)
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()

            # Predict
            buffer.seek(0)
            predictions = model_handler.predict(buffer)

            return render_template(
                "result.html",
                original_filename=file.filename,
                image_b64=img_b64,
                predictions=predictions
            )
        else:
            flash("Invalid file type. Allowed: PNG, JPG, JPEG")

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

