import streamlit as st


def init_ui(get_answer: callable) -> None:
    st.set_page_config(page_title="Instagram Review Insight LLM", page_icon="🧠")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Title and clear button
    st.title("💬 Instagram Review Insight LLM")
    st.write("")
    if st.button("Clear Messages"):
        st.session_state.messages = []

    with st.container():
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Display chat input
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            st.write("Analyzing reviews...")
            response = get_answer(prompt)
            st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
    return