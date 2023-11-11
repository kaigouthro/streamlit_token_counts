import tiktoken
import streamlit as st

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        st.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|im_start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|im_start|>assistant<|im_sep|>
    return num_tokens



def main():
    st.title("Message Token Counter")
    st.write("Count the number of tokens used by a list of messages.")
    
    messages = st.session_state.get('mesageslist', [
            {
                "role": "system",
                "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": "New synergies will help drive top-line growth.",
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": "Things working well together will increase revenue.",
            },
            {
                "role": "system",
                "name": "example_user",
                "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",
            },
            {
                "role": "system",
                "name": "example_assistant",
                "content": "Let's talk later when we're less busy about how to do better.",
            },
            {
                "role": "user",
                "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",
            }
        ])
    
    # Create message input
    with st.form(key="message_input"):
        st.header("Create Message")
        
        role = st.selectbox("Role", ["user", "assistant", "system"])
        name = st.text_input("Name (optional)")
        content = st.text_area("Content")
        
        submit_button = st.form_submit_button("Create")
        
        if submit_button:
            message = {"role": role, "content": content}
            if name:
                message["name"] = name
            messages.append(message)
    
    # Remove message
    if st.button("Remove Last Message") and messages:
        messages.pop()
    
    st.session_state['mesageslist'] = messages
    
    # Display chat messages
    st.subheader("Chat Messages")
    for message in messages:
        with st.chat_message(f"{message['role']}"):           
            if "name" not in message:
                st.write(message['content'])
            else:
                st.write(f"{message['name']   }:")
                st.write(f"{message['content']}" )
        
    selected_model = st.selectbox("Model", ["gpt-3.5-turbo", "gpt-4"])
    
    # Display the token count based on the selected model
    token_count = num_tokens_from_messages(messages, model=selected_model)
    st.markdown(f"# Token Count for Model '{selected_model}': {token_count}")
    
if __name__ == "__main__":
    main()
