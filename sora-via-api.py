import streamlit as st
from openai import OpenAI
import os
import requests
from PIL import Image
import io
import base64
from dotenv import load_dotenv
import json
from datetime import datetime, date
import zipfile
import pandas as pd
import csv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .image-container {
        border: 2px solid #e1e1e1;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(145deg, #f0f0f0, #ffffff);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .settings-box {
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


def get_api_key():
    """Get OpenAI API key from environment or .env file"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("⚠️ OpenAI API key not found! Please set OPENAI_API_KEY in your environment or .env file")
        st.info("Create a .env file in your project directory with: OPENAI_API_KEY=your_api_key_here")
        return None
    return api_key

def init_spending_database(db_file="spending_tracker.csv"):
    """Initialize spending database CSV file if it doesn't exist"""
    if not os.path.exists(db_file):
        with open(db_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['date', 'model', 'images_generated', 'cost_per_image', 'total_cost', 'prompt_snippet', 'timestamp'])
    return db_file

def add_spending_record(db_file, model, images_count, cost_per_image, prompt):
    """Add a spending record to the CSV database"""
    try:
        today = date.today().isoformat()
        total_cost = cost_per_image * images_count
        prompt_snippet = prompt[:50] + "..." if len(prompt) > 50 else prompt
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(db_file, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([today, model, images_count, cost_per_image, total_cost, prompt_snippet, timestamp])
        
        return total_cost
    except Exception as e:
        st.error(f"Error saving spending record: {str(e)}")
        return 0

def load_spending_data(db_file):
    """Load spending data from CSV"""
    try:
        if os.path.exists(db_file):
            df = pd.read_csv(db_file)
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df
        else:
            return pd.DataFrame(columns=['date', 'model', 'images_generated', 'cost_per_image', 'total_cost', 'prompt_snippet', 'timestamp'])
    except Exception as e:
        st.error(f"Error loading spending data: {str(e)}")
        return pd.DataFrame(columns=['date', 'model', 'images_generated', 'cost_per_image', 'total_cost', 'prompt_snippet', 'timestamp'])

def get_daily_spending(df, target_date=None):
    """Get spending for a specific date (default: today)"""
    if target_date is None:
        target_date = date.today()
    
    daily_data = df[df['date'] == target_date]
    return daily_data['total_cost'].sum()

def get_spending_summary(df):
    """Get comprehensive spending summary"""
    if df.empty:
        return {
            'today': 0,
            'this_week': 0,
            'this_month': 0,
            'total': 0,
            'daily_breakdown': pd.DataFrame()
        }
    
    today = date.today()
    
    # Today's spending
    today_spending = get_daily_spending(df, today)
    
    # This week's spending (last 7 days)
    week_data = df[df['date'] >= today - pd.Timedelta(days=7)]
    week_spending = week_data['total_cost'].sum()
    
    # This month's spending
    month_data = df[df['date'].apply(lambda x: x.month == today.month and x.year == today.year)]
    month_spending = month_data['total_cost'].sum()
    
    # Total spending
    total_spending = df['total_cost'].sum()
    
    # Daily breakdown for the last 30 days
    daily_breakdown = df.groupby('date')['total_cost'].sum().reset_index()
    daily_breakdown = daily_breakdown.sort_values('date', ascending=False).head(30)
    
    return {
        'today': today_spending,
        'this_week': week_spending,
        'this_month': month_spending,
        'total': total_spending,
        'daily_breakdown': daily_breakdown
    }

def download_image(image_url, filename):
    """Download image from URL and return PIL Image"""
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Error downloading image: {str(e)}")
        return None

def save_image_locally(image, filename, output_dir):
    """Save image to local directory"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        return filepath
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None

def create_download_link(image, filename):
    """Create download link for image"""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'


def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 AI Image Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by OpenAI's Latest Models")
    
    # Check API key
    api_key = get_api_key()
    if not api_key:
        return
    
    # Initialize spending database
    db_file = init_spending_database()
    spending_df = load_spending_data(db_file)
    spending_summary = get_spending_summary(spending_df)
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        st.info("Make sure you have the latest OpenAI library: pip install openai>=1.0.0")
        return
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model = st.selectbox(
            "Model",
            ["gpt-image-1", "dall-e-3"],
            index=0,
            help="GPT-Image-1: Latest model with superior quality and instruction following. DALL-E 3: Previous generation model."
        )
        
        # Image size and quality options
        if model == "gpt-image-1":
            # GPT-Image-1 has simplified parameters - size/quality handled internally
            n_images = st.slider("Number of Images", 1, 4, 1, help="GPT-Image-1 can generate up to 4 images per request")
            st.info("📋 GPT-Image-1 automatically optimizes size and quality based on your prompt")
            cost_per_image = 0.040 # Standard pricing for GPT-Image-1
        elif model == "dall-e-3":
            size_options = ["1024x1024", "1792x1024", "1024x1792"]
            size = st.selectbox("Image Size", size_options, index=0)
            n_images = 1
            
            quality = st.selectbox(
                "Quality",
                ["standard", "hd"],
                index=0,
                help="HD quality provides finer details but costs more"
            )
            
            style = st.selectbox(
                "Style",
                ["vivid", "natural"],
                index=0,
                help="Vivid: hyper-real and dramatic, Natural: more natural and less hyper-real"
            )
            if size == "1024x1024":
                cost_per_image = 0.040 if quality == "standard" else 0.080
            else:  # Higher resolution
                cost_per_image = 0.080 if quality == "standard" else 0.120
            
        st.markdown("---")
        
        # Daily Budget & Spending (Sidebar)
        st.subheader("💰 Daily Budget")
        
        # Daily budget setting
        daily_budget = st.number_input(
            "Daily Budget ($)",
            min_value=0.0,
            value=5.0,
            step=0.50,
            help="Set your daily spending limit. A value of $0.00 is considered infinite."
        )
        
        # Today's spending only
        today_spent = spending_summary['today']

        # Custom display for infinite budget
        if daily_budget == 0:
            st.metric("Today's Spending", f"${today_spent:.3f}", "Infinite remaining")
            st.success("✅ Budget is set to infinite.")
        else:
            remaining_budget = max(0, daily_budget - today_spent)
            st.metric("Today's Spending", f"${today_spent:.3f}", f"${remaining_budget:.3f} remaining")
            
            # Budget status indicator
            if today_spent >= daily_budget:
                st.error("⚠️ Daily budget exceeded!")
            elif today_spent >= daily_budget * 0.8:
                st.warning("⚠️ 80% of daily budget used")
            else:
                st.success(f"✅ {((today_spent/daily_budget)*100):.1f}% of budget used")
        
        st.markdown("---")
        
        # Output settings
        st.subheader("📁 Output Settings")
        
        # Default output directory
        default_output_dir = st.text_input(
            "Default Output Directory",
            value="generated_images",
            help="Directory where images will be saved locally"
        )
        
        # Auto-save option
        auto_save = st.checkbox(
            "Auto-save images locally",
            value=True,
            help="Automatically save generated images to the output directory"
        )
        
        # Filename format
        filename_format = st.selectbox(
            "Filename Format",
            [
                "timestamp_prompt",
                "prompt_timestamp",
                "custom_prefix",
                "sequential_number"
            ],
            index=0
        )
        
        if filename_format == "custom_prefix":
            custom_prefix = st.text_input("Custom Prefix", value="ai_art")
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings"):
            
            # Session settings
            st.subheader("📊 Session & Data")
            if 'generation_count' not in st.session_state:
                st.session_state.generation_count = 0
            
            st.info(f"Images this session: {st.session_state.generation_count}")
            
            if st.button("Reset Session Counter"):
                st.session_state.generation_count = 0
                st.success("Session counter reset!")
                
            # Export spending data
            if not spending_df.empty:
                st.markdown("**Export Data:**")
                csv_data = spending_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Spending CSV",
                    data=csv_data,
                    file_name=f"spending_data_{date.today()}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🎨 Generate Images", "📊 Spending Dashboard", "🖼️ Image Gallery"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Prompt input
            st.markdown('<div class="settings-box">', unsafe_allow_html=True)
            st.subheader("✍️ Image Prompt")
            
            # Predefined prompts for inspiration
            prompt_examples = [
                "A serene mountain landscape at sunset with a crystal-clear lake reflecting the orange and pink sky, professional landscape photography",
                "A futuristic cityscape with flying cars and neon lights, cyberpunk style, highly detailed architecture",
                "A magical forest with glowing mushrooms and fairy lights, fantasy art style with ethereal lighting",
                "An abstract geometric pattern with vibrant gradients and flowing shapes, modern digital art",
                "A vintage café in Paris with people sitting outside, impressionist painting style with warm lighting",
                "A majestic dragon perched on a castle tower during golden hour, epic fantasy illustration",
                "A minimalist modern kitchen with marble countertops and natural lighting, architectural photography",
                "A bustling Tokyo street at night with colorful signs and reflections, neon photography style"
            ]
            
            example_prompt = st.selectbox(
                "💡 Example Prompts (optional)",
                [""] + prompt_examples,
                index=0,
                help="Select an example prompt or write your own below"
            )
            
            prompt = st.text_area(
                "Enter your image prompt:",
                value=example_prompt,
                height=100,
                placeholder="Describe the image you want to generate in detail...",
                help="Be specific and descriptive for better results. Include style, mood, colors, and composition details."
            )
            
            # Prompt enhancement options
            col_a, col_b = st.columns(2)
            with col_a:
                add_style = st.checkbox("Add artistic style", help="Append artistic style keywords")
                if add_style:
                    style_options = [
                        "photorealistic", "oil painting", "watercolor", "digital art",
                        "3D render", "sketch", "pop art", "minimalist", "baroque",
                        "impressionist", "surreal", "anime style", "comic book style"
                    ]
                    selected_style = st.selectbox("Select style:", style_options)
                    if selected_style and prompt:
                        prompt = f"{prompt}, {selected_style}"
            
            with col_b:
                add_quality = st.checkbox("Add quality enhancers", help="Add quality-improving keywords")
                if add_quality and prompt:
                    prompt = f"{prompt}, highly detailed, professional photography, 8K resolution"
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Generate button with budget check
            budget_check_passed = True
            
            # Only perform budget checks if a positive budget is set
            if daily_budget > 0:
                if spending_summary['today'] >= daily_budget:
                    st.error("⚠️ Daily budget exceeded! Adjust your budget in the sidebar or wait until tomorrow.")
                    budget_check_passed = False
                
                estimated_cost = cost_per_image * n_images
                if spending_summary['today'] + estimated_cost > daily_budget:
                    st.warning(f"⚠️ This generation (${estimated_cost:.3f}) would exceed your daily budget!")
                    budget_override = st.checkbox("Generate anyway (override budget)", key="budget_override")
                    budget_check_passed = budget_override
            
            if st.button("🎨 Generate Images", type="primary", use_container_width=True, disabled=not budget_check_passed):
                if not prompt:
                    st.error("Please enter a prompt!")
                    return
                
                with st.spinner("🎨 Generating your masterpiece..."):
                    try:
                        # Generate images using the correct API structure
                        if model == "gpt-image-1":
                            # GPT-Image-1 uses simplified API structure
                            response = client.images.generate(
                                model=model,
                                prompt=prompt,
                                n=n_images
                            )
                        else:
                            # DALL-E 3 uses the traditional parameters
                            response = client.images.generate(
                                model=model,
                                prompt=prompt,
                                size=size,
                                quality=quality,
                                style=style,
                                n=n_images
                            )
                        
                        # Update session counter
                        st.session_state.generation_count += len(response.data)
                        
                        # Store generated images in session state for better handling
                        if 'generated_images' not in st.session_state:
                            st.session_state.generated_images = []
                        
                        # Process images and add to session state
                        current_batch = []
                        for idx, image_data in enumerate(response.data):
                            image = None
                            try:
                                # GPT-Image-1 and newer models return base64 encoded images by default
                                if hasattr(image_data, 'b64_json') and image_data.b64_json:
                                    # Decode base64 image
                                    image_bytes = base64.b64decode(image_data.b64_json)
                                    image = Image.open(io.BytesIO(image_bytes))
                                    
                                elif hasattr(image_data, 'url') and image_data.url:
                                    # Handle URL format (fallback for older models)
                                    image = download_image(image_data.url, f"generated_image_{idx}.png")
                                    
                                if image:
                                    # Generate filename
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
                                    
                                    if filename_format == "timestamp_prompt":
                                        filename = f"{timestamp}_{safe_prompt}_{idx}.png"
                                    elif filename_format == "prompt_timestamp":
                                        filename = f"{safe_prompt}_{timestamp}_{idx}.png"
                                    elif filename_format == "custom_prefix":
                                        filename = f"{custom_prefix}_{timestamp}_{idx+1}.png"
                                    else:  # sequential_number
                                        filename = f"generated_image_{st.session_state.generation_count - len(response.data) + idx + 1}.png"
                                    
                                    # Auto-save if enabled
                                    if auto_save:
                                        filepath = save_image_locally(image, filename, default_output_dir)
                                        if filepath:
                                            st.success(f"💾 Saved: {filepath}")
                                    
                                    current_batch.append({
                                        'image': image,
                                        'filename': filename,
                                        'prompt': prompt,
                                        'timestamp': timestamp
                                    })
                                else:
                                    st.error(f"Unable to process image {idx + 1}")
                                    
                            except Exception as img_error:
                                st.error(f"Error processing image {idx + 1}: {str(img_error)}")
                        
                        # Add current batch to session state
                        st.session_state.generated_images.extend(current_batch)
                        
                        # Record spending
                        actual_cost = add_spending_record(db_file, model, len(current_batch), cost_per_image, prompt)
                        
                        # Display success message with cost
                        st.success(f"✅ Generated {len(current_batch)} image(s)! Cost: ${actual_cost:.3f}")
                        
                        # Refresh spending data
                        spending_df = load_spending_data(db_file)
                        spending_summary = get_spending_summary(spending_df)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating image: {str(e)}")
                        st.info("💡 Troubleshooting tips:")
                        st.info("- Make sure you have the latest OpenAI library: `pip install openai>=1.0.0`")
                        st.info("- Check your API key and billing status")
                        st.info("- Try simplifying your prompt")
                        st.info(f"- Error details: {type(e).__name__}")
                
        with col2:
            st.markdown('<div class="settings-box">', unsafe_allow_html=True)
            st.subheader("📊 Generation Info")
            
            if prompt:
                st.write("**Current Prompt:**")
                st.write(f'"{prompt}"')
                st.write(f"**Estimated tokens:** {len(prompt.split())}")
            
            st.write(f"**Model:** {model}")
            if model == "dall-e-3":
                st.write(f"**Size:** {size}")
                st.write(f"**Quality:** {quality}")
                st.write(f"**Style:** {style}")
            else:  # gpt-image-1
                st.write(f"**Size:** Auto-optimized")
                st.write(f"**Quality:** Auto-optimized")
            st.write(f"**Images to generate:** {n_images}")
            
            estimated_cost = cost_per_image * n_images
            st.write(f"**Estimated cost:** ${estimated_cost:.3f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Tips section
            with st.expander("💡 Prompting Tips"):
                st.markdown("""
                **For better results:**
                - Be specific about style, colors, and composition
                - Include lighting details (e.g., "soft morning light")
                - Mention camera angles or perspectives
                - Add mood or atmosphere descriptions
                - Use artist names for specific styles
                - Include quality keywords: "highly detailed", "8K", "professional"
                
                **Example good prompt:**
                "A majestic lion in golden savanna grass during sunset, dramatic lighting, wildlife photography style, shallow depth of field, warm orange tones, highly detailed"
                """)
            
            # Recent outputs section (if any images were generated)
            if 'generation_count' in st.session_state and st.session_state.generation_count > 0:
                with st.expander("📈 Session Statistics"):
                    st.write(f"Images generated: {st.session_state.generation_count}")
                    if auto_save and os.path.exists(default_output_dir):
                        files = [f for f in os.listdir(default_output_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        st.write(f"Images saved locally: {len(files)}")
    
    with tab2:
        # Spending Dashboard
        st.header("💰 Spending Dashboard")
        
        # Overview metrics
        col_dash_1, col_dash_2, col_dash_3, col_dash_4 = st.columns(4)
        
        with col_dash_1:
            st.metric("Today", f"${spending_summary['today']:.3f}")
        with col_dash_2:
            st.metric("This Week", f"${spending_summary['this_week']:.3f}")
        with col_dash_3:
            st.metric("This Month", f"${spending_summary['this_month']:.3f}")
        with col_dash_4:
            st.metric("All Time", f"${spending_summary['total']:.3f}")
        
        # Daily breakdown chart
        if not spending_summary['daily_breakdown'].empty:
            st.subheader("📈 Daily Spending Trend (Last 30 Days)")
            chart_df = spending_summary['daily_breakdown'].copy()
            chart_df['date'] = pd.to_datetime(chart_df['date'])
            st.line_chart(data=chart_df.set_index('date'), y='total_cost', use_container_width=True)
        
        # Detailed analytics
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Model usage breakdown
            if not spending_df.empty:
                st.subheader("🤖 Model Usage")
                model_stats = spending_df.groupby('model').agg({
                    'total_cost': 'sum',
                    'images_generated': 'sum'
                }).reset_index()
                st.dataframe(
                    model_stats,
                    use_container_width=True,
                    column_config={
                        'model': 'Model',
                        'total_cost': st.column_config.NumberColumn('Total Cost', format="$%.3f"),
                        'images_generated': 'Images Generated'
                    }
                )
        
        with col_right:
            # Top spending days
            if not spending_df.empty:
                st.subheader("💸 Top Spending Days")
                top_days = spending_df.groupby('date')['total_cost'].sum().reset_index()
                top_days = top_days.sort_values('total_cost', ascending=False).head(10)
                st.dataframe(
                    top_days,
                    use_container_width=True,
                    column_config={
                        'date': 'Date',
                        'total_cost': st.column_config.NumberColumn('Spending', format="$%.3f")
                    }
                )
        
        # Recent transactions
        if not spending_df.empty:
            st.subheader("📋 Recent Transactions")
            recent_transactions = spending_df.sort_values('timestamp', ascending=False).head(20)
            st.dataframe(
                recent_transactions[['date', 'model', 'images_generated', 'total_cost', 'prompt_snippet', 'timestamp']],
                use_container_width=True,
                column_config={
                    'date': 'Date',
                    'model': 'Model',
                    'images_generated': 'Images',
                    'total_cost': st.column_config.NumberColumn('Cost', format="$%.3f"),
                    'prompt_snippet': 'Prompt',
                    'timestamp': 'Time'
                }
            )
    
    with tab3:
        # Display generated images section
        if 'generated_images' in st.session_state and st.session_state.generated_images:
            st.header("🖼️ Your Generated Images")
            
            # Show recent images first
            recent_images = st.session_state.generated_images[-12:]  # Show last 12 images
            
            if len(recent_images) > 1:
                cols = st.columns(min(3, len(recent_images)))
            else:
                cols = [st]  # Use single column for one image
            
            for idx, image_data in enumerate(recent_images):
                # Use columns for multiple images, direct streamlit for single image
                if len(recent_images) > 1:
                    container = cols[idx % 3]
                else:
                    container = st
                
                # Check if the data is a dictionary with a valid image before displaying
                if isinstance(image_data, dict) and 'image' in image_data:
                    # Display image
                    container.markdown('<div class="image-container">', unsafe_allow_html=True)
                    container.image(
                        image_data['image'],
                        caption=f"Generated: {image_data['timestamp']}",
                        use_container_width=True
                    )
                    
                    # Show prompt snippet
                    prompt_snippet = image_data['prompt'][:50] + "..." if len(image_data['prompt']) > 50 else image_data['prompt']
                    container.caption(f"Prompt: {prompt_snippet}")
                    
                    # Download link
                    container.markdown(
                        create_download_link(image_data['image'], image_data['filename']),
                        unsafe_allow_html=True
                    )
                    container.markdown('</div>', unsafe_allow_html=True)
                # The problematic 'else' block has been removed to prevent the AttributeError
            
            # Clear images button
            if st.button("🗑️ Clear Image History"):
                st.session_state.generated_images = []
                st.success("Image history cleared!")
                st.rerun()
        else:
            st.info("No images generated yet. Go to the 'Generate Images' tab to create your first image!")


if __name__ == "__main__":
    main()
