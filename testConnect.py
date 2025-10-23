from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
hf_token = "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"  # replace with real token
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",          # :white_check_mark: correct for this model
    temperature=0.7,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)
chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("What is AI")
print(response.content)





