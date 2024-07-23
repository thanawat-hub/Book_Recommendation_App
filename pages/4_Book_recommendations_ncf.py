# ปกติใน streamlit มันรันจากบนลงล่าง ทำให้เปลืองทรัยากร วิธีแก้คือ
# 1. cache cross-session เข้าถึง X ได้พร้อมกัน เช่น datasets (ค่าเหมือนกันหมด) ก็เก็บที่ level นี้ วิธีคือ ผ่าน decorator @st.cache แล้วไปดูว่าจะใช้ อะไร กับ session อะไร
# 2. session (single session) มันอิสระต่อกัน เช่นไปดูค่าเดียวกัน session1 (มองเป็น user) ไว้ปรับแต่งของตัวเอง session2 จะไม่เห็นของ session1

import streamlit as st

# -----
import torch

from util.All_model.model import GMF, MLP, NCF

from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data_with_LabelEncoder():
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    # Fit and transform user and item IDs for create new column name
    st.session_state['ratings_df']['user_id'] = user_encoder.fit_transform(st.session_state['ratings_df']['User-ID'])
    st.session_state['ratings_df']['item_id'] = item_encoder.fit_transform(st.session_state['ratings_df']['ISBN'])

    # Create mappings
    user_id_map = {id: idx for idx, id in enumerate(st.session_state['ratings_df']['User-ID'].unique())}
    item_id_map = {id: idx for idx, id in enumerate(st.session_state['ratings_df']['ISBN'].unique())}

    st.session_state['ratings_df']['user_idx'] = st.session_state['ratings_df']['User-ID'].map(user_id_map)
    st.session_state['ratings_df']['item_idx'] = st.session_state['ratings_df']['ISBN'].map(item_id_map)

    return user_id_map, item_id_map, st.session_state['ratings_df'], item_encoder# for training then eval using just user_id


def display_top_books(isbn_list, n_show_book):
    books_df = st.session_state['books_df']
    displayed_books = 0  # Track the number of displayed books

    for isbn in isbn_list:
        if displayed_books >= n_show_book:#that have isbn
            break  # Stop after displaying top 3 books

        book = books_df[books_df['ISBN'] == isbn]
        if not book.empty:
            book_title = book['Book-Title'].values[0]
            book_author = book['Book-Author'].values[0]
            image_url = book['Image-URL-L'].values[0]

            st.subheader(f"Book name: {book_title}")
            st.text(f"Author: {book_author}")
            st.image(image_url, caption=f"ISBN: {isbn}")

            displayed_books += 1
        else:
            # st.write(f"No information found for ISBN: {isbn}")
            pass
def book_recsys():
    # ตรงนี้คือ session level -> คือหลังบ้าน ทำการโหลด df มารอไว้ละ
    if 'datasets' in st.session_state:
        if 'user_id_map' not in st.session_state:
            with st.spinner('convert ratings_df to train NCF...'):
                st.session_state['user_id_map'], st.session_state['item_id_map'], st.session_state['ratings_df'], st.session_state['item_encoder'] = load_data_with_LabelEncoder()

    st.subheader("Dataframe to use to input to NCF")
    st.dataframe(st.session_state['ratings_df'])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_users = len(st.session_state['user_id_map'])
    num_items = len(st.session_state['item_id_map'])
    embedding_size = 64
    hidden_layers = [128, 64, 32]

    # Initialize GMF model
    gmf_model = GMF(num_users, num_items, embedding_size).to(device)

    # Initialize MLP model
    mlp_model = MLP(num_users, num_items, embedding_size, hidden_layers).to(device)

    model = NCF(gmf_model, mlp_model)
    model.load_state_dict(torch.load("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/util/All_model/model.pth"))#from "NFC_10Epoch_no_add_feature[0].ipynb"

    # Preprocessing the top 64 books
    book_ratings = st.session_state['ratings_df'].groupby('item_idx')['Book-Rating'].mean().reset_index()
    book_ratings = book_ratings.sort_values(by='Book-Rating', ascending=False)
    top_64_books = book_ratings.head(64)
    # st.write(top_64_books)

    # print(type(st.session_state['user_id_map']))#dictonary
    first_fifty = dict(list(st.session_state['user_id_map'].items())[10:50])

    # st.subheader('Select User ID')
    user_id = st.selectbox(
        "Select User ID",
        first_fifty,#show just 100 user to rec book
        placeholder="Select User ID...",
    )

    user_idx = st.session_state['user_id_map'][user_id]
    user_id_tensor = torch.LongTensor([user_idx] * 64).to(device)

    top_64_books = top_64_books['item_idx'].tolist()
    item_ids_tensor = torch.LongTensor(top_64_books).to(device)

    # st.write("User ID Tensor:", user_id_tensor)
    # st.write("Top 10 Books Tensor:", item_ids_tensor)

    predictions = model(user_id_tensor, item_ids_tensor)

    indexed_predictions = [(idx, pred) for idx, pred in enumerate(predictions)]

    # Sort the indexed predictions by the prediction values in descending order
    sorted_predictions = sorted(indexed_predictions, key=lambda x: x[1], reverse=True)
    top_3_indices = [idx for idx, _ in sorted_predictions[:10]]#เอามาก่อน10 แล้วเข้า fn ให้ plot ที่มีแค่3
    top_3_book_isbns_id = [item_ids_tensor[idx].item() for idx in top_3_indices]

    top_3_book_isbns_id_convert2_ISBN = st.session_state['item_encoder'].inverse_transform(top_3_book_isbns_id)#convert กลับเป็นISBNS book

    n_show_book = st.number_input("Insert a number to show books", min_value=3, max_value=20, value=5)
    display_top_books(top_3_book_isbns_id_convert2_ISBN, n_show_book)

if __name__ == '__main__':

    st.title('Book recommended')
    st.header('Using Neural Collaborative Filtering')
    # st.image("https://qph.cf2.quoracdn.net/main-qimg-8b7db09f0026709117d0369aeaaee360-lq")#url
    st.image("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/images/ncf_user_id.png", caption='Neural Collaborative Filtering')

    if 'ratings_df' not in st.session_state:
        st.warning("Go to page EDA_Book_datasets first")
    else:
        book_recsys()
