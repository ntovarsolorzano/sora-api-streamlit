# **Streamlit AI Image Generator**

This is a simple and intuitive web application built with Streamlit that allows users to generate images using the OpenAI API. It provides a user-friendly interface to configure prompts, model parameters, and track spending.

## **Features**

* 🖼️ **AI Image Generation:** Generate images using the latest OpenAI models like gpt-image-1 and dall-e-3.  
* ✍️ **Prompt Customization:** Enhance prompts with predefined examples, artistic styles, and quality enhancers.  
* 💰 **Spending Tracker:** Monitor daily, weekly, and monthly costs associated with image generation.  
* 📊 **Usage Dashboard:** View a daily spending trend and model usage statistics.  
* 📁 **Local Storage:** Automatically save generated images to a local directory with customizable naming conventions.  
* 📥 **Easy Downloads:** Download individual images directly from the ggeneratedallery.

## **Prerequisites**

* Python 3.8+  
* An OpenAI API Key

## **Installation & Setup**

1. **Clone the repository or download the script:**  
   `git clone https://github.com/ntovarsolorzano/sora-api-streamlit.git`
   
   `cd sora-api-streamlit`  

3. **Install the required Python libraries:**  
   `pip install streamlit openai pandas python-dotenv Pillow requests`

4. **Set up your OpenAI API Key:**  
   Create a file named .env in the same directory as the script with the following content:  
   `OPENAI_API_KEY="your\_api\_key\_here"`

   Replace "your\_api\_key\_here" with your actual API key.

## **Usage**

Run the Streamlit application from your terminal:  
`streamlit run sora-via-api.py`  

This will open the app in your default web browser.
