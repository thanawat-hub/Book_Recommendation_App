import time

import streamlit as st

import re
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer #tokenizes a collection of words extracted from a text doc


@st.cache_data
def load_data_book_df_pre(): #จะเอาแค่ book_df ที่มี Rating ไม่ใช่ค่า 0 
    books_df4content_base = st.session_state['books_df'].merge(st.session_state['ratings_df'],on="ISBN")
    books_df4content_base.dropna(inplace=True)
    books_df4content_base.reset_index(drop=True, inplace=True)
    books_df4content_base.drop(columns=["ISBN", "Year-Of-Publication"], axis=1, inplace=True)
    books_df4content_base.drop(index=books_df4content_base[books_df4content_base["Book-Rating"] == 0].index, inplace=True)
    books_df4content_base["Book-Title"] = books_df4content_base["Book-Title"].apply(lambda x: re.sub(r"[\W_]+", " ", x).strip())
    
    return books_df4content_base


# Function to randomly select and remove an element from the array
def choice_and_pop(arr, rng):
    # Randomly select an element
    idx = rng.choice(len(arr))
    element = arr[idx]
    # Remove the element by creating a new array without the selected element
    arr = np.delete(arr, idx)
    return element, arr

def content_based(bookTitle):
    bookTitle = str(bookTitle)

    if bookTitle in st.session_state['books_df4content_base']["Book-Title"].values:

        rating_count = pd.DataFrame(st.session_state['books_df4content_base']["Book-Title"].value_counts())
        rare_books = rating_count[rating_count["count"] <= 200].index#กำหนดไปเลยว่า ถ้า rating น้อยกว่า 200 คือหายาก ก็ไม่แนะนำ
        common_books = st.session_state['books_df4content_base'][~st.session_state['books_df4content_base']["Book-Title"].isin(rare_books)]
        if bookTitle in rare_books:
            most_common = pd.Series(common_books["Book-Title"].unique()).sample(10).values
            st.write("No Recommendations for this Book ☹️")
            st.write("You May Try:")


            # Create a sample NumPy array
            array = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

            # Initialize the random number generator
            rng = np.random.default_rng()

            # Randomly select and remove elements
            random_element1, array = choice_and_pop(array, rng)
            random_element2, array = choice_and_pop(array, rng)
            random_element3, array = choice_and_pop(array, rng)

            st.write("{}".format(most_common[random_element1]), "\n")
            st.write("{}".format(most_common[random_element2]), "\n")
            st.write("{}".format(most_common[random_element3]), "\n")
        else:
            common_books = common_books.drop_duplicates(subset=["Book-Title"])#ลบชื่อที่ซ้ำ
            common_books.reset_index(inplace=True)
            common_books["index"] = [i for i in range(common_books.shape[0])]#ทำ index ใหม่
            targets = ["Book-Title", "Book-Author", "Publisher"]
            common_books["all_features"] = [" ".join(common_books[targets].iloc[i,].values) for i in
                                            range(common_books[targets].shape[0])]#สร้าง column all_features รวม ค่าจาก target column


            with st.expander("explaination"):
                st.subheader("Dataframe preprocess from original data to use")
                st.dataframe(st.session_state['books_df4content_base'].head())

                st.subheader("Dataframe to use for content-base filtering")
                st.dataframe(common_books)

            vectorizer = CountVectorizer()
            common_booksVector = vectorizer.fit_transform(common_books["all_features"])# แปลง ค่าใน column all_feature โดยใช้ vectorizer เป็น vector
            similarity = cosine_similarity(common_booksVector)
            index = common_books[common_books["Book-Title"] == bookTitle]["index"].values[0]#หาว่าชื่อที่ input เข้ามา ตรงกับ book-title index อะไร
            similar_books = list(enumerate(similarity[index]))#ได้ similar_books ที่ใกล้กับ index

            similar_booksSorted = sorted(similar_books, key=lambda x: x[1], reverse=True)[1:6]# Sorted อันดับ 1-6
            books = []
            for i in range(len(similar_booksSorted)):
                books.append(common_books[common_books["index"] == similar_booksSorted[i][0]]["Book-Title"].item())
            for i, book in enumerate(books):
                url = common_books.loc[common_books["Book-Title"] == book, "Image-URL-L"][:1].values[0]
                st.write(f"Book: {book}")
                st.image(url, caption=f"RATING {round(st.session_state['books_df4content_base'][st.session_state['books_df4content_base']["Book-Title"] == books[i]]["Book-Rating"].mean(), 1)}")
    else:
        st.subheader(f"❌ COULD NOT FIND '{bookTitle}' BOOK IN DATABASES❌")

def book_recsys_content_base():
    if "books_df4content_base" not in st.session_state:
        st.session_state['books_df4content_base'] = load_data_book_df_pre()

    st.subheader("Type your favorite book you like I will find that similar book")

    # # #https://discuss.streamlit.io/t/can-i-add-to-a-selectbox-an-other-option-where-the-user-can-add-his-own-answer/28525/3
    # # -------------------------------
    #Extract unique book titles
    book_titles = st.session_state['books_df4content_base']['Book-Title'].unique()
    # Set the seed for reproducibility

    input_number = st.number_input("Insert a number to random name of book in datasets", min_value=1)
    np.random.seed(int(input_number))
    random_elements = np.random.choice(book_titles, size=3)

    options = [f"{title}" for title in random_elements] + ["Another_option"]
    selection = st.selectbox("Select option", options=options)

    # Create text input for user entry
    if selection == "Another_option":
        otherOption = st.text_input("Enter your other Book...")

    # Just to show the selected option
    if selection != "Another_option":
        # st.info(f":white_check_mark: The selected option is {selection} ")
        if selection is not None:
            content_based(selection)
    else:
        # st.info(f":white_check_mark: The written option is {otherOption} ")
        if otherOption is not None:
            content_based(otherOption)

if __name__ == '__main__':

    st.title('Book recommended system')
    st.image("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/images/content-base_filtering_half.png", caption='Collaborative Filtering')

    if 'ratings_df' not in st.session_state:
        st.warning("Go to page EDA_Book_datasets first")
    else:
        book_recsys_content_base()
