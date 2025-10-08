import os
# from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=('AIzaSyBQT4mPl5nCUh3jq79sd8M06pxGPQwzfsc'))
response = llm.invoke("Hello, can you respond?")
print(response.content)