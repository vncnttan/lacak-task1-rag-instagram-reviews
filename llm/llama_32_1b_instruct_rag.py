import time
import openai 

# Set the BASE URL to the local server
openai.base_url = "http://localhost:1234/v1/"
openai.api_key = "NoNeed"

def get_llm_response(prompt: str, rag_info: str, max_tokens:int=500, temperature:float=0) -> str:
    # Get response from the LLM hosted on the local server LM Studio
    try:
        print("Generating response...")
        # Create responses using the instruct model
        response = openai.chat.completions.create(
            model="llama-3.2-1b-instruct",
            messages=[
                { "role": "system", "content": f"""
                Analyze these user reviews to answer: {prompt}
                    Consider:
                    - Common patterns across reviews
                    - Sentiment trends
                    - Specific user pain points
                    - Changes in feedback over time

                Reviews: {rag_info}
                """ },
                {"role": "system", "content": "Answer in paragraph form. Our app is Instagram." },
                { "role": "user", "content": prompt }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {str(e)}")
        return None