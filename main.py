from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from transformers import pipeline
import os
import shutil
import tempfile

app = FastAPI()

# Configurar el middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def obtenerClasificacion(path):
    pipe = pipeline("image-classification", model="MiguelCalderon/google-vit-base-patch16-224-OrganicAndInorganicWaste")
    prediction = pipe(path)
    class_label = prediction[0]['label']
    probability = prediction[0]['score']
    return class_label, probability

@app.get("/hola-mundo")
async def hola_mundo():
    return {"mensaje": "Hola Mundo"}

@app.post("/clasificarImagen")
async def clasificarImagen(image: UploadFile = File(...)):
    if not (image.filename.endswith(".svg") or image.filename.endswith(".png") or image.filename.endswith(
            ".jpeg") or image.filename.endswith(".jpg")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            shutil.copyfileobj(image.file, temp_image)
            temp_image_path = temp_image.name

        class_label, _ = obtenerClasificacion(temp_image_path)  # Obtener la clasificación
        #base_path = r"C:\apitesis\clasification"
        #class_folder_path = os.path.join(base_path, class_label) # Crear la carpeta para la clasificación si no existe
        #os.makedirs(class_folder_path, exist_ok=True)
        #new_image_path = os.path.join(class_folder_path, image.filename)# Mover la imagen al directorio correspondiente
        #shutil.move(temp_image_path, new_image_path)

        return {"label": class_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))