# Azure AI Foundry SDK

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

try:
    # hoặc dùng DefaultAzureCredential()
    api_key = "api_key"
    key_credential = AzureKeyCredential(api_key)
    
    # connect to the project
    project_endpoint = "https://ngothilinhau-ai-foundry-resource.services.ai.azure.com/api/projects/ngothilinhau-ai-foundry-project." # Azure AI Foundry project endpoint
    project_client = AIProjectClient(            
            # credential=DefaultAzureCredential(),
            credential=key_credential,
            endpoint=project_endpoint,
        )
    
    # Get a chat client
    chat_client = project_client.inference.get_azure_openai_client(api_version="2024-10-21")
    
    # Get a chat completion based on a user-provided prompt
    user_prompt = input("Enter a question:")
    
    response = chat_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(response.choices[0].message.content)

except Exception as ex:
    print(ex)