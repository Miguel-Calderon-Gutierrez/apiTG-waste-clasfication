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
    pipe = pipeline("image-classification", model="MiguelCalderon/google-vit-base-patch16-224-Waste-O-I-classification")
    prediction = pipe(path)
    class_label = prediction[0]['label']
    probability = prediction[0]['score']
    return class_label, probability

@app.get("/getDataPrueba")
def prueba():
    return {"label": "O", "score": "https://t1.uc.ltmcdn.com/es/posts/8/8/1/que_se_tira_en_el_contenedor_organico_33188_600.jpg"}

@app.post("/clasificarImagen")
async def clasificarImagen(image: UploadFile = File(...)):
    if not (image.filename.endswith(".svg") or image.filename.endswith(".png") or image.filename.endswith(
            ".jpeg") or image.filename.endswith(".jpg")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            shutil.copyfileobj(image.file, temp_image)
            temp_image_path = temp_image.name

        class_label, score = obtenerClasificacion(temp_image_path)  # Obtener la clasificación
        #base_path = r"C:\apitesis\clasification"
        #class_folder_path = os.path.join(base_path, class_label) # Crear la carpeta para la clasificación si no existe
        #os.makedirs(class_folder_path, exist_ok=True)
        #new_image_path = os.path.join(class_folder_path, image.filename)# Mover la imagen al directorio correspondiente
        #shutil.move(temp_image_path, new_image_path)

        return {"label": class_label, "score": score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clasificarYguadarImagen")
async def clasificarYguardarImagen(image: UploadFile = File(...)):
    if not (image.filename.endswith(".svg") or image.filename.endswith(".png") or image.filename.endswith(
            ".jpeg") or image.filename.endswith(".jpg")):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_image:
            shutil.copyfileobj(image.file, temp_image)
            temp_image_path = temp_image.name

        class_label, score = obtenerClasificacion(temp_image_path)
        base_path = r"C:\apitesis\clasification"
        class_folder_path = os.path.join(base_path, class_label)
        os.makedirs(class_folder_path, exist_ok=True)
        new_image_path = os.path.join(class_folder_path, image.filename)

        shutil.move(temp_image_path, new_image_path)

        return {"label": class_label, "score": score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))