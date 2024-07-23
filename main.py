# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import streamlit as st

import base64

# from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():

    # เราสามารถให้มัน load datasets ให้เสร็จตรงนี้เลย แล้วทำ LabelEncoder กับงาน CF
    # books_df = pd.read_csv("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/Books.csv", low_memory=False, dtype={'ISBN': str})
    # ratings_df = pd.read_csv("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/Ratings.csv", low_memory=False, dtype={'User-ID': str,'ISBN': str})#for show
    # users_df = pd.read_csv("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/Users.csv", low_memory=False)

    books_df = pd.read_csv("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/Books.csv")
    ratings_df = pd.read_csv("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/Ratings.csv")
    users_df = pd.read_csv("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/Users.csv")

    return books_df.drop(['Image-URL-S', 'Image-URL-M'], axis=1), ratings_df, users_df# original
    # user_encoder = LabelEncoder()
    # item_encoder = LabelEncoder()
    # # Fit and transform user and item IDs
    # ratings_df['user_id'] = user_encoder.fit_transform(ratings_df['User-ID'])
    # ratings_df['item_id'] = item_encoder.fit_transform(ratings_df['ISBN'])
    #
    # # Create mappings
    # user_id_map = {id: idx for idx, id in enumerate(ratings_df['User-ID'].unique())}
    # item_id_map = {id: idx for idx, id in enumerate(ratings_df['ISBN'].unique())}
    #
    # ratings_df['user_idx'] = ratings_df['User-ID'].map(user_id_map)
    # ratings_df['item_idx'] = ratings_df['ISBN'].map(item_id_map)
    #
    # return user_id_map, item_id_map, books_df, ratings_df, users_df# for training


def print_welcome(name):
    return f"""
    ## :notebook_with_decorative_cover: :blue[Welcome, {name}!]

    ### Guidelines for using this app:
    - Use the sidebar to navigate through different sections my journey for do this following pages number.
        - [0] welcome -> main for suggest and loading a datasets (in backend streamlit)
        - [1] EDA to all datasets from kaggle and find a limitation of this kaggle NCF in => Book_recommended.
        - [2] Book recommendations content-base search similarity with your favorite book.
        - [3] Book recommendations using collaborative filtering (CF) with Singular Value Decomposition (SVD). CF often uses Matrix Factorization (MF) techniques.
        - [4] Book recommendations [to train pure user and item embedding into NCF in [0]ipynb] then predict in this pages using "user_id" as inputs. Limitation which is more vulnerable to the well-known cold-start problem. However, in next we'll modify the architecture to include additional user and item features [3] 
        - [Bonus] Rating visualization with Dimensional Reduction (DR)
    - Enjoy exploring new books! 📚 (This datasets is yearbooks from the 1990-2000 era:face_with_monocle:)
    """
# - reference
# kaggle
# https: // www.kaggle.com / code / ahb1104 / neural - collaborative - filtering -> book
# https://www.kaggle.com/code/oyounis/ncf-recommender-system  # NCF-Recommender-System-with-PyTorch:-A-Deep-Dive-%F0%9F%9A%80 -> movielen

# :text:
# https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/

def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    return base64_image

def add_logo(image_path):
    base64_image = get_base64_image(image_path)
    # background-image: url(https://img.freepik.com/free-photo/book-composition-with-open-book_23-2147690555.jpg);# if using url_image it is this replace in line 60 ({base64_image})
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url("data:image/png;base64,{base64_image}");
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
                background-size: 200px; /* Adjust size as needed */
                # opacity: 0.5; /* Adjust transparency (0.0 to 1.0) */
            }}
            [data-testid="stSidebarNav"]::before {{
                content: "Navigate";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 30px;
                position: relative;
                top: 100px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Main function to run the Streamlit app
if __name__ == '__main__':
    # Set the title of the app
    st.title("Book Recommendation App")
    st.write(print_welcome("To Book Recommendation"))
    add_logo("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/book.jpg")

    if 'datasets' and 'ratings_df' not in st.session_state:#กรณี ของ collaborative
        with st.spinner('Loading data...'):
            # st.session_state['user_id_map'], st.session_state['item_id_map'], st.session_state['books_df'], st.session_state['ratings_df'], st.session_state['users_df'] = load_data()
            st.session_state['books_df'], st.session_state['ratings_df'], st.session_state['users_df'] = load_data()
            st.session_state['books_df']['ISBN'] = st.session_state['books_df']['ISBN'].astype(str)#to handle this string in column

            st.session_state['datasets'] = ":black_joker:"

    st.write(st.session_state['datasets'])
