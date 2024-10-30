import os
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 0,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

from ultralytics import YOLO
import os
import glob
from IPython.display import Image, display
model = YOLO("best.pt")
pic = "Recipe-Ingredients-CN-4/test/images/155_png_jpg.rf.03807a35b88f5727d160dcb786d75276.jpg"
#pic = "https://img.wongnai.com/p/1920x0/2017/12/24/552c29cacda543f3bd9786cf464b08a2.jpg"
results = model.predict(source=pic, conf=0.35, save=True)

detected_items = {}
for result in results:
    for box in result.boxes:
        cls = box.cls
        class_name = model.names[int(cls)]
        if class_name in detected_items:
            detected_items[class_name] += 1
        else:
            detected_items[class_name] = 1

print(detected_items)


model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  safety_settings=safety_settings,
  generation_config=generation_config,
  system_instruction="I want a food recipe based on the following ingredients:{}. Please format the recipe with the following sections:Ingredients: List all ingredients required for the recipe.Instructions: Provide clear step-by-step instructions for preparing the recipe.Optional Tips: Include any optional tips for variation or improvement of the recipe. แปลเป็นไทย",
)

chat_session = model.start_chat(
    history=[]
)

print()
print("Planning a menu based on the ingredients list...")

while True:

    user_input = (str(detected_items))
    print()

    response = chat_session.send_message(user_input)

    model_response = response.text

    print(f'menu: {model_response}')
    print()

    chat_session.history.append({"role": "user", "parts": [user_input]})
    chat_session.history.append({"role": "model", "parts": [model_response]})
    break
