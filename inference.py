import torch
import numpy as np
from stable_baselines3 import PPO
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI()

# Mount the static files (HTML, CSS, JS for the game)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

class Pipeline:
    def __init__(self):
        # Load the model
        self.model = PPO.load("model_RL_Cosmic_latest", device='cpu', custom_objects={"map_location": torch.device('cpu')})

    def __call__(self, inputs):
        obs_data = inputs.get("observation")
        if obs_data is None:
            return {"error": "No observation provided"}

        obs = self.preprocess_observation(obs_data)
        if isinstance(obs, dict) and "error" in obs:
            return obs

        action, _ = self.model.predict(obs, deterministic=True)
        return {"action": action.tolist()}

    def preprocess_observation(self, obs_data):
        try:
            if isinstance(obs_data, str):
                img_bytes = base64.b64decode(obs_data)
                img = Image.open(io.BytesIO(img_bytes)).convert('L').resize((100, 150))
                img_array = np.array(img, dtype=np.float32) / 255.0
                obs = np.stack([img_array for _ in range(8)], axis=0)
            elif isinstance(obs_data, list):
                obs = np.array(obs_data, dtype=np.float32)
                if obs.shape != (8, 150, 100):
                    return {"error": f"Incorrect observation shape: expected (8, 150, 100), got {obs.shape}"}
            else:
                return {"error": "Invalid observation format"}

            obs = obs[np.newaxis, ...]
            return obs
        except Exception as e:
            return {"error": f"Failed to process observation: {e}"}

pipeline = Pipeline()

class ObservationInput(BaseModel):
    observation: str

@app.post("/predict")
def predict(input_data: ObservationInput):
    result = pipeline({"observation": input_data.observation})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
