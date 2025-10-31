import os, io, json, random, requests
from dataclasses import dataclass, asdict
from typing import List
from PIL import Image, ImageDraw
import streamlit as st
from dotenv import load_dotenv
import numpy as np

# ------------------------------------
# Load environment variables
# ------------------------------------
load_dotenv()
HF_KEY = os.getenv("HUGGINGFACE_API_KEY")

# ------------------------------------
# Config
# ------------------------------------
# List of models to try in order
MODEL_OPTIONS = [
    "runwayml/stable-diffusion-v1-5",
    "CompVis/stable-diffusion-v1-4",
    "black-forest-labs/FLUX.1-schnell" # This one might require more permissions
]

CONFIG = {
    "COHERENCE_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "IMAGE_CACHE_DIR": "generated_images"
}
os.makedirs(CONFIG["IMAGE_CACHE_DIR"], exist_ok=True)

# ------------------------------------
# Dataclasses
# ------------------------------------
@dataclass
class CreativeBrief:
    raw_brief: str
    target_audience: str = ""
    tone: str = ""
    emotion: str = ""
    platform: str = ""
    key_message: str = ""
    constraints: str = "" # Added from v1 for completeness

@dataclass
class CopyImagePair:
    copy_text: str
    image_path: str
    prompt: str
    coherence_score: float = 0.0

# ------------------------------------
# Helpers
# ------------------------------------
def save_image_bytes(name: str, data: bytes) -> str:
    """Saves image bytes to the cache directory."""
    path = os.path.join(CONFIG["IMAGE_CACHE_DIR"], name)
    with open(path, "wb") as f:
        f.write(data)
    return path

def placeholder_image_bytes(msg="Image gen failed") -> bytes:
    """Creates a grey placeholder image with text."""
    img = Image.new("RGB", (512, 512), (230,230,230))
    d = ImageDraw.Draw(img)
    try:
        # Try to load a font for better text
        from PIL import ImageFont
        # You might need to ensure 'arial.ttf' is available or use a default
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except (ImportError, OSError):
            font = ImageFont.load_default() # Fallback to default
        d.text((50,240), msg, fill=(20,20,20), font=font)
    except (ImportError, OSError):
        # Basic fallback if font loading fails
        d.text((50,240), msg, fill=(20,20,20))
    b = io.BytesIO(); img.save(b, "PNG")
    return b.getvalue()

def display_image(path, caption=None):
    """Safely displays an image from a path in Streamlit."""
    try:
        st.image(Image.open(path), caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display image: {path} (Error: {e})")

# ------------------------------------
# Copy generation (from v1)
# ------------------------------------
def generate_copies(cb: CreativeBrief, n=5) -> List[str]:
    """Generates a list of ad copy strings based on templates."""
    base = cb.key_message
    audience = cb.target_audience or "your target audience"
    tone = cb.tone or "a unique"
    emotion = cb.emotion or "interest"

    # Using the more extensive template list from v1
    templates = [
        f"{base} — designed for {audience}. {tone.capitalize()} and bold.",
        f"{base}. Perfect for {audience} seeking {emotion}.",
        f"{base} — feel the {emotion} now!",
        f"{tone.capitalize()} vibe: {base}. Tap to explore!",
        f"{base}. Made for {audience}. Experience {emotion}.",
        f"Discover {base} — crafted for {audience} who love {emotion}.",
        f"{base}: The {tone} choice for {audience}.",
        f"Unleash {emotion} with {base}. For {audience}.",
        f"{base} — where {tone} meets {emotion} for {audience}.",
        f"Get ready for {base}! {tone.capitalize()} style for {audience}."
    ]
    
    if n > len(templates):
        # Repeat templates if more copies are requested than available
        templates = templates * (n // len(templates) + 1)
    
    return random.sample(templates, n)

# ------------------------------------
# Craft image prompt (from v1)
# ------------------------------------
def craft_image_prompt(copy: str, cb: CreativeBrief) -> str:
    """Creates a descriptive image prompt from the brief and copy."""
    return (
        f"Professional advertisement for: {copy}. "
        f"Style: {cb.tone} tone, evoking {cb.emotion}. "
        f"Target audience: {cb.target_audience}. "
        f"Platform: {cb.platform}. "
        "High quality, vibrant colors, commercial aesthetic."
    )

# ------------------------------------
# Alternative Image Generation (from v2, renamed)
# ------------------------------------
def generate_image_alternative(prompt: str) -> bytes:
    """Create programmatic images as fallback when API fails"""
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Add gradient based on emotion keywords
    if any(word in prompt.lower() for word in ['energy', 'vibrant', 'bold', 'excitement']):
        # Vibrant gradient
        for i in range(height):
            r = int(255 * (i / height))
            g = int(128 + 127 * (i / height))
            b = int(255 * (1 - i / height))
            draw.line([(0, i), (width, i)], fill=(r, g, b))
    elif any(word in prompt.lower() for word in ['calm', 'wellness', 'natural', 'peaceful']):
        # Calm gradient
        for i in range(height):
            r = int(100 + 100 * (i / height))
            g = int(200 + 55 * (i / height))
            b = int(150 + 100 * (i / height))
            draw.line([(0, i), (width, i)], fill=(r, g, b))
    else:
        # Neutral gradient
        for i in range(height):
            shade = int(200 + 55 * (i / height))
            draw.line([(0, i), (width, i)], fill=(shade, shade, shade))
    
    # Add text overlay
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
        words = prompt.split()[:5] # First 5 words as "headline"
        headline = " ".join(words)
        draw.text((50, 200), headline, fill=(0, 0, 0), font=font)
        draw.text((50, 230), "Advertisement Preview", fill=(100, 100, 100), font=font)
    except (ImportError, OSError):
        draw.text((50, 200), "Generated Ad", fill=(0, 0, 0))
    
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()

# ------------------------------------
# Image generation (from v2, modified)
# ------------------------------------
def generate_image(prompt: str) -> bytes:
    """Tries multiple HF models and falls back to programmatic image."""
    if not HF_KEY:
        st.error("❌ Missing Hugging Face API key")
        return generate_image_alternative(prompt)
    
    for model in MODEL_OPTIONS:
        st.info(f"🔄 Attempting to generate image with: {model}")
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {HF_KEY}"}
        payload = {"inputs": prompt}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                st.success(f"✅ Image generated successfully with {model}!")
                return response.content
            elif response.status_code == 503:
                st.warning(f"⏳ Model {model} is loading, trying next...")
                continue # Try next model
            elif response.status_code == 404:
                st.warning(f"❌ Model {model} not found or not accessible, trying next...")
                continue # Try next model
            elif response.status_code == 401:
                st.error("❌ Invalid API token or insufficient permissions.")
                # This is a fatal key error, stop trying
                return generate_image_alternative("API token permission issue")
            else:
                st.warning(f"API Error {response.status_code} with {model}: {response.text[:100]}...")
                continue # Try next model
                
        except requests.exceptions.Timeout:
            st.warning(f"⏰ Request timeout with {model}, trying next...")
            continue # Try next model
        except Exception as e:
            st.error(f"❌ Unexpected error with {model}: {str(e)}")
            continue # Try next model
    
    # If all models fail
    st.error("❌ All API models failed. Using fallback programmatic image.")
    return generate_image_alternative(prompt)

# ------------------------------------
# Coherence Evaluation (from v1)
# ------------------------------------
def get_coherence_score(copy_text: str, image_prompt: str) -> float:
    """Get semantic coherence score with fallback methods"""
    
    # Method 1: Try Hugging Face API
    if HF_KEY:
        try:
            url = f"https://api-inference.huggingface.co/models/{CONFIG['COHERENCE_MODEL']}"
            headers = {"Authorization": f"Bearer {HF_KEY}"}
            payload = {
                "inputs": {
                    "source_sentence": copy_text,
                    "sentences": [image_prompt]
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    score = float(result[0])
                    return min(max(score, 0.0), 1.0) # Clamp score between 0 and 1
            else:
                st.warning(f"Coherence API failed ({response.status_code}), using fallback score.")
        except Exception as e:
            st.warning(f"Coherence check failed ({e}), using fallback score.")
    
    # Method 2: Keyword overlap (Fallback)
    copy_words = set(copy_text.lower().split())
    prompt_words = set(image_prompt.lower().split())
    common_words = copy_words.intersection(prompt_words)
    
    if len(copy_words) > 0:
        # v1's more advanced fallback score
        base_score = len(common_words) / len(copy_words)
        length_ratio = min(len(copy_text) / len(image_prompt), 1.0) if len(image_prompt) > 0 else 0.5
        enhanced_score = (base_score * 0.7) + (length_ratio * 0.3)
        return round(enhanced_score, 2)
    
    # Absolute fallback
    return round(random.uniform(0.5, 0.8), 2)

# ------------------------------------
# A/B Test Recommendations (from v1)
# ------------------------------------
def generate_ab_test_recommendations(pairs: List[CopyImagePair]) -> List[dict]:
    """Generate intelligent A/B test recommendations"""
    recommendations = []
    
    if len(pairs) < 2:
        return [{"title": "Need more variants", "description": "Generate at least 2 variants for A/B testing", "variant_a": "N/A", "variant_b": "N/A", "hypothesis": "N/A"}]
    
    sorted_pairs = sorted(pairs, key=lambda x: x.coherence_score, reverse=True)
    
    # Test 1: Top 2 coherence battle
    top1, top2 = sorted_pairs[0], sorted_pairs[1]
    recommendations.append({
        "title": "🏆 High Coherence Showdown",
        "description": f"Test your two best-performing variants head-to-head",
        "variant_a": f"Ad 1 (Score: {top1.coherence_score:.2f}): '{top1.copy_text}'",
        "variant_b": f"Ad 2 (Score: {top2.coherence_score:.2f}): '{top2.copy_text}'",
        "hypothesis": "Both have strong message-visual alignment; test which emotional angle resonates better"
    })
    
    # Test 2: High vs Low coherence
    if len(pairs) >= 3:
        lowest = sorted_pairs[-1]
        recommendations.append({
            "title": "⚡ Safe vs Bold Approach",
            "description": "Compare a reliable option against an unconventional one",
            "variant_a": f"Safe Choice (Ad 1, Score: {top1.coherence_score:.2f}): '{top1.copy_text}'",
            "variant_b": f"Bold Choice (Ad {len(pairs)}, Score: {lowest.coherence_score:.2f}): '{lowest.copy_text}'",
            "hypothesis": "Test whether audience prefers coherent messaging or unconventional approaches"
        })
    
    return recommendations

# ------------------------------------
# Streamlit UI (Merged)
# ------------------------------------
def run_app():
    st.set_page_config(page_title="Ad Agent — Copy & Image Generator", layout="wide")
    st.title("🎯 Ad Copy & Image Generator with Coherence Evaluation")
    
    # --- Sidebar (Merged) ---
    st.sidebar.header("Configuration")
    st.sidebar.info(f"Coherence Model: {CONFIG['COHERENCE_MODEL']}")
    
    with st.sidebar.expander("Image Models (Tried in order)"):
        for model in MODEL_OPTIONS:
            st.write(f"- {model}")
    
    if not HF_KEY:
        st.sidebar.error("❌ HUGGINGFACE_API_KEY not found in .env file")
        st.warning("Please add your Hugging Face API key to a .env file. Using fallback images only.")
    else:
        st.sidebar.success("✅ API Key found!")
    
    with st.sidebar.expander("🔑 API Token Help"):
        st.write("""
        **If images aren't generating, your token might be read-only.**
        
        This app will try multiple models to find one that works. If all fail, it will create a placeholder image.
        
        **Solutions:**
        1. Ensure your token has 'write' permissions on Hugging Face.
        2. Check that the models in the list are accessible to you.
        """)
    
    # --- Main interface (from v1) ---
    brief = st.text_area(
        "Enter campaign brief:",
        value="Promote our new sugar-free energy drink with natural ingredients",
        height=100,
    )
    
    st.subheader("Define Ad Parameters")
    col1, col2 = st.columns(2)
    with col1:
        target_audience = st.text_input("Target Audience", "Health-conscious millennials")
        tone = st.text_input("Tone", "energetic and authentic")
    with col2:
        emotion = st.text_input("Emotion", "vitality and wellness")
        platform = st.text_input("Platform", "Instagram Stories")
    
    st.subheader("Generation Settings")
    n_var = st.slider("Number of ad copies", 2, 8, 4)
    top_k = st.slider("Top results to show", 1, min(n_var, 5), min(3, n_var))
    
    if st.button("🚀 Generate Ads", type="primary"):
        if not brief.strip():
            st.error("Please enter a campaign brief")
            return
            
        # Create CreativeBrief
        cb = CreativeBrief(
            raw_brief=brief,
            key_message=brief,
            target_audience=target_audience,
            tone=tone,
            emotion=emotion,
            platform=platform
        )
        
        pairs = []
        
        # Generate copies
        with st.spinner("📝 Generating ad copies..."):
            copies = generate_copies(cb, n_var)
            st.success(f"✅ Generated {len(copies)} ad copies")
            
            with st.expander("View Generated Copies"):
                for i, copy in enumerate(copies, 1):
                    st.write(f"**{i}.** {copy}")
        
        # Generate images
        st.info("🖼️ Generating images...")
        progress_bar = st.progress(0)
        
        for i, copy in enumerate(copies):
            with st.spinner(f"Generating image {i+1}/{len(copies)}..."):
                prompt = craft_image_prompt(copy, cb)
                
                # Use the robust generate_image function
                image_bytes = generate_image(prompt) 
                
                filename = f"ad_{i}_{hash(copy) % 10000}.png"
                path = save_image_bytes(filename, image_bytes)
                pairs.append(CopyImagePair(copy_text=copy, image_path=path, prompt=prompt))
                progress_bar.progress((i + 1) / len(copies))
        
        # Evaluate coherence (from v1)
        with st.spinner("🔍 Evaluating copy-image coherence..."):
            for i, pair in enumerate(pairs):
                pair.coherence_score = get_coherence_score(pair.copy_text, pair.prompt)
        
        st.success("🎉 All steps completed!")
        
        # --- Display results (from v1) ---
        best_pairs = sorted(pairs, key=lambda x: x.coherence_score, reverse=True)[:top_k]
        
        st.header("🏆 Top Performing Combinations")
        for i, pair in enumerate(best_pairs, 1):
            st.subheader(f"Variant {i} (Coherence Score: {pair.coherence_score:.2f})")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                display_image(pair.image_path, f"Variant {i}")
            with col2:
                st.write(f"**Ad Copy:** {pair.copy_text}")
                st.write(f"**Target:** {cb.target_audience} | **Tone:** {cb.tone}")
                st.write(f"**Emotion:** {cb.emotion} | **Platform:** {cb.platform}")
                with st.expander("📋 View Image Prompt"):
                    st.code(pair.prompt)
            
            st.write("---")
        
        # A/B Test Recommendations (from v1)
        st.header("🔬 A/B Testing Recommendations")
        recommendations = generate_ab_test_recommendations(pairs)
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"Recommendation {i}: {rec['title']}"):
                st.write(rec['description'])
                st.write("**Test Variants:**")
                st.write(f"- **Variant A:** {rec['variant_a']}")
                st.write(f"- **Variant B:** {rec['variant_b']}")
                st.write(f"**Hypothesis:** {rec['hypothesis']}")
        
        # Performance Summary (from v1)
        st.header("📊 Performance Summary")
        if pairs: # Avoid division by zero if list is empty
            avg_coherence = sum(p.coherence_score for p in pairs) / len(pairs)
            max_coherence = max(p.coherence_score for p in pairs)
            min_coherence = min(p.coherence_score for p in pairs)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Variants", len(pairs))
            col2.metric("Average Coherence", f"{avg_coherence:.2f}")
            col3.metric("Best Coherence", f"{max_coherence:.2f}")
            col4.metric("Worst Coherence", f"{min_coherence:.2f}")
            
            # Coherence distribution chart (from v1)
            st.subheader("Coherence Score Distribution")
            chart_data = {f"Variant {i+1}": p.coherence_score for i, p in enumerate(sorted(pairs, key=lambda x: x.coherence_score, reverse=True))}
            st.bar_chart(chart_data)
        else:
            st.warning("No pairs were generated.")

if __name__ == "__main__":
    run_app()
