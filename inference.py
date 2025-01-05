import torch
import numpy as np
from stable_baselines3 import PPO
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

# Initialize FastAPI
app = FastAPI()

# Define the /predict endpoint before mounting StaticFiles
class Pipeline:
    def __init__(self):
        # Load the model
        print("Loading the model...")
        self.model = PPO.load("model_RL_Cosmic_latest", device='cpu', custom_objects={"map_location": torch.device('cpu')})
        print("Model loaded successfully.")

    def __call__(self, inputs):
        print("Pipeline called with inputs:", inputs)
        obs_data = inputs.get("observation")
        if obs_data is None:
            print("No observation provided.")
            return {"error": "No observation provided"}

        obs = self.preprocess_observation(obs_data)
        if isinstance(obs, dict) and "error" in obs:
            print("Error in preprocessing observation:", obs["error"])
            return obs

        print("Observation preprocessed successfully. Shape:", obs.shape)
        action, _ = self.model.predict(obs, deterministic=True)
        print("Model predicted action:", action)
        return {"action": action.tolist()}

    def preprocess_observation(self, obs_data):
        print("Preprocessing observation...")
        try:
            if isinstance(obs_data, list) and len(obs_data) == 8:
                frames = []
                for idx, encoded_frame in enumerate(obs_data):
                    print(f"Decoding frame {idx+1}/8")
                    img_bytes = base64.b64decode(encoded_frame.split(",")[1])
                    img = Image.open(io.BytesIO(img_bytes)).convert('L').resize((100, 150))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    frames.append(img_array)
                obs = np.stack(frames, axis=0)  # Shape: (8, 150, 100)
                obs = obs[np.newaxis, ...]  # Add batch dimension
                return obs
            else:
                error_msg = "Invalid observation format or insufficient frames"
                print(error_msg)
                return {"error": error_msg}
        except Exception as e:
            error_msg = f"Failed to process observation: {e}"
            print(error_msg)
            return {"error": error_msg}

pipeline = Pipeline()

class ObservationInput(BaseModel):
    observation: List[str]

@app.post("/predict")
def predict(input_data: ObservationInput):
    print("Received a prediction request.")
    print("Received observation data type:", type(input_data.observation))
    result = pipeline({"observation": input_data.observation})
    if "error" in result:
        print("Error in pipeline:", result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    print("Returning result:", result)
    return result

# Mount the static files after defining other routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (including POST)
    allow_headers=["*"],
)
