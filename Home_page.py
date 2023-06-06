
import streamlit as st
import joblib
import pandas as pd


df = pd.read_csv('make_up.csv')
l = list(df['text2'])

idx_to_str = {0:'Negative', 1:'Positive'}

st.set_page_config(page_title="Emotion prediction")
st.write("# Hi, Dear merchant👋")
st.markdown(
    """
    Want to gain insights about the emotion trend of your customers? 
    Just input a customer's comment into the box(in English format).
    """
)
st.title('Emotion prediction')
content = st.text_area("Please enter a customer's English comment (only one comment can be predicted at a time)", key = 0)



if st.button('Run', key=1):
    with st.spinner('The system is loading and inferring model, please wait...'):
        @st.cache(hash_funcs={"MyUnhashableClass": lambda _: None}, allow_output_mutation=True)
        def load_model():
            return joblib.load('./RFModel.pkl'), joblib.load('./Vectorizer.pkl')

        model, vectorizer = load_model()

    l.append(content)
    embed = vectorizer.fit_transform(l)
    e = list(embed.toarray())
    p = model.predict(e)
    result = p[-1]
    output = idx_to_str[result]
    st.write('The predicted emotion is', output)
    st.success('Model loading and inference successful!')

