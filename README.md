# AdSparkAssignment
# Problem 2: Multi-Modal Campaign Asset Generator

## Statement:

Build an agent that takes a campaign brief, generates multiple ad copy variations, creates image generation prompts for each, calls image generation API, evaluates copy-image coherence, and packages best combinations with A/B test recommendations.

## Ad Copy & Image Generator with Coherence Scoring

This is a Streamlit web application that serves as an "Ad Agent." It takes a creative brief from a user and generates multiple ad variants, each consisting of AI-generated ad copy and a corresponding AI-generated image.

The application then scores each ad pair for "coherence" (how well the text and image match) and presents the top-performing combinations, a performance summary, and A/B testing recommendations.

(Note: The screenshot is a representative example of a Streamlit app with this layout.)

## ✨ Core Features

Creative Brief Input: Users can define a campaign's key message, target audience, tone, emotion, and platform.

Ad Copy Generation: Generates multiple ad copy variants based on dynamic templates.

AI-Powered Image Generation: For each ad copy, a unique image is generated using the Hugging Face Inference API.

Robust Fallback System:

Multi-Model Retry: The app tries a list of image generation models in order (e.g., Stable Diffusion v1.5, v1.4, FLUX.1-schnell).

Error Handling: It intelligently handles common API errors like model loading (503), timeouts, or permission issues.

Programmatic Fallback: If all API models fail, it generates a clean, programmatic placeholder image with a gradient and text instead of crashing.

Coherence Scoring: Uses a sentence-transformer model (via API) to score the semantic similarity between the ad copy and the image prompt. A keyword-based fallback is used if the API fails.

A/B Test Recommendations: Automatically suggests A/B tests (e.g., "High Coherence Showdown," "Safe vs. Bold") based on the coherence scores.

Performance Dashboard: Displays the top-ranked ads, key metrics (avg/best/worst scores), and a bar chart visualizing the coherence distribution of all variants.

## 🚀 How to Run

1. Prerequisites

Python 3.8+

A Hugging Face account and API Key.

2. Setup & Installation

Clone or Download:
Get the assignment_8.py file and place it in a new project directory.

Create requirements.txt:
Create a file named requirements.txt in the same directory and add the contents from the file provided below.

Create a Virtual Environment (Recommended):

# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate



Install Dependencies:

pip install -r requirements.txt



Create .env File:
Create a file named .env in the same directory. This will store your API key securely.

Inside the .env file, add your key:

HUGGINGFACE_API_KEY="hf_YOUR_API_KEY_GOES_HERE" (do remember to create a write action API token)



Important: Your Hugging Face API Key must have write permissions to be able to use the inference models. You can create one from your Hugging Face settings.

3. Launch the App

With your virtual environment active, run the following command in your terminal:

streamlit run assignment_8.py



This will automatically open the application in your web browser.

## ⚙️ How It Works (Workflow)

Input: The user fills out the brief (key message, audience, tone, etc.) in the Streamlit UI.

Copy Gen: The app generates n ad copy strings using generate_copies().

Prompt Crafting: For each copy, a detailed image prompt is built using craft_image_prompt().

Image Gen: The app calls generate_image() for each prompt. This function loops through the MODEL_OPTIONS list, attempting to get an image from the Hugging Face API.

Fallback: If the API calls fail (e.g., model is loading, 503 error), generate_image_alternative() is triggered, creating a local placeholder image.

Scoring: The app calls get_coherence_score() for each copy/prompt pair. This tries the sentence-transformers API first, falling back to a keyword-overlap score if needed.

Display: The generated pairs are sorted by their coherence score. The top k results are displayed, along with A/B test ideas and the summary dashboard.
