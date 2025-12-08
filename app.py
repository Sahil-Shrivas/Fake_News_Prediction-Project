# import streamlit as st
# import pickle
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()

# vector_form = pickle.load(open('vector.pkl', 'rb'))
# load_model = pickle.load(open('model.pkl', 'rb'))

# def stemming(content):
#     con=re.sub('[^a-zA-Z]', ' ', content)
#     con=con.lower()
#     con=con.split()
#     con=[port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con=' '.join(con)
#     return con

# def fake_news(news):
#     news=stemming(news)
#     input_data=[news]
#     vector_form1=vector_form.transform(input_data)
#     prediction = load_model.predict(vector_form1)
#     return prediction



# if __name__ == '__main__':
#     st.title('Fake News Classification app ')
#     st.subheader("Input the News content below")
#     sentence = st.text_area("Enter your news content here", "",height=200)
#     predict_btt = st.button("predict")
#     if predict_btt:
#         prediction_class=fake_news(sentence)
#         print(prediction_class)
#         if prediction_class == [0]:
#             st.success('Reliable')
#         if prediction_class == [1]:
#             st.warning('Unreliable')















# import streamlit as st
# import pickle
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer

# port_stem = PorterStemmer()
# vectorization = TfidfVectorizer()

# vector_form = pickle.load(open('vector.pkl', 'rb'))
# load_model = pickle.load(open('model.pkl', 'rb'))

# def stemming(content):
#     con = re.sub('[^a-zA-Z]', ' ', content)
#     con = con.lower()
#     con = con.split()
#     con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
#     con = ' '.join(con)
#     return con

# def fake_news(news):
#     news = stemming(news)
#     input_data = [news]
#     vector_form1 = vector_form.transform(input_data)
#     prediction = load_model.predict(vector_form1)
#     return prediction


# # ------------------- UI -------------------
# if __name__ == '__main__':

#     st.markdown(
#         """
#         <h1 style='text-align:center; color:#4A90E2;'>
#             📰 Fake News Classification App
#         </h1>
#         <p style='text-align:center; font-size:18px; color:gray;'>
#             Enter any news content below and analyze its reliability.
#         </p>
#         <hr>
#         """,
#         unsafe_allow_html=True
#     )

#     st.subheader("📝 Enter the News Content")
#     sentence = st.text_area("Write or paste the news text below:", "", height=200)

#     predict_btt = st.button("🔍 Analyze News", use_container_width=True)

#     if predict_btt:
#         if sentence.strip() == "":
#             st.error("Please enter some text first.")
#         else:
#             result = int(fake_news(sentence)[0])   # FIXED HERE

#             if result == 0:
#                 st.success("✅ **Reliable News**")
#             else:
#                 st.warning("⚠️ **Unreliable News**")






















import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Load NLTK stopwords
# -----------------------------
nltk.download('stopwords')
stop_words = stopwords.words('english')

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

# -----------------------------
# Preprocessing Function
# -----------------------------
port_stem = PorterStemmer()

def preprocess_text(content):
    content = re.sub('[^a-zA-Z]', ' ', content)  # Keep only letters
    content = content.lower()
    content = content.split()

    # Remove stopwords + apply stemming
    content = [port_stem.stem(word) for word in content if word not in stop_words]

    return ' '.join(content)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_news(news):
    processed = preprocess_text(news)
    
    vector_data = vector_form.transform([processed])  # Vectorize
    prediction = load_model.predict(vector_data)      # Predict
    
    return int(prediction[0]), processed


# -----------------------------
# Streamlit UI
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2E86C1;'>
        📰 Fake News Classification App
    </h1>
    <p style='text-align:center; color:gray; font-size:18px;'>
        Paste any news content below and analyze if it's trustworthy.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

st.subheader("📝 Enter News Content")
news_text = st.text_area("Paste your news article here:", height=200)

analyze_btn = st.button("🔍 Analyze News", use_container_width=True)

if analyze_btn:
    if news_text.strip() == "":
        st.error("Please enter some news text before analyzing.")
    else:
        prediction, processed = predict_news(news_text)

        # Show processed version for debugging (you can remove this)
        # st.write("Processed Text:", processed)

        if prediction == 0:
            st.success("✅ **Reliable News** — The content seems trustworthy.")
        else:
            st.warning("⚠️ **Unreliable News** — This content may not be trustworthy.")
