from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
import joblib

app = FastAPI()

oneri_model = joblib.load("oneri_model_onehot.pkl")
label_encoder = joblib.load("label_encoder.pkl")

class KullaniciVerisi(BaseModel):
    diyet: str

@app.get("/")
def root():
    return {"message": "Öneri API çalışıyor!"}

@app.post("/recommend/")
def yemek_onerisi(veri: KullaniciVerisi):
    try:
        response = requests.get("https://fitplate-backend.onrender.com/last-3-meals/")
        response.raise_for_status()
        ogun_listesi = response.json()["ogunler"]
    except Exception as e:
        return {"hata": "Tahmin API'ye ulaşılamıyor.", "detay": str(e)}

    if len(ogun_listesi) < 3:
        return {
            "mesaj": "Öneri için en az 3 öğün gerekli.",
            "ogun_sayisi": len(ogun_listesi),
            "ogunler": ogun_listesi
        }

    toplam = {"protein": 0, "karbonhidrat": 0, "yağ": 0}
    for ogun in ogun_listesi:
        for k in toplam:
            toplam[k] += ogun["besin"][k]

    ortalama = {k: round(v / 3, 2) for k, v in toplam.items()}
    df = pd.DataFrame([{**ortalama, "diyet": veri.diyet}])
    tahmin = oneri_model.predict(df)[0]
    yemek = label_encoder.inverse_transform([tahmin])[0]

    return {
        "önerilen_yemek": yemek,
        "gerekçe": {
            "diyet_tipi": veri.diyet,
            "son_3_ögün_ortalama": ortalama,
            "toplam_besin_degerleri": toplam
        },
        "not": "Bu öneri son 3 öğününüze ve diyet tercihinize göre yapılmıştır."
    }
