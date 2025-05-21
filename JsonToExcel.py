import json
import pandas as pd
import os

json_file = os.path.join(os.path.dirname(__file__), "pose_data.json")
excel_file = os.path.join(os.path.dirname(__file__), "pose_data.xlsx")

with open(json_file, "r") as f:
    data_list = json.load(f)

rows = []

for frame_idx, frame in enumerate(data_list):
    for person_idx, person_data in enumerate(frame):
        rows.append({
            "Frame": frame_idx + 1,  
            "Persona": person_idx + 1, 
            "Distancia entre ojos": person_data["distancia_ojos"],  
            "Distancia entre hombres": person_data["distancia_hombros"], 
            "Distancia entre cadera": person_data["distancia_cadera"] 
        })

df = pd.DataFrame(rows)

df.to_excel(excel_file, index=False)
