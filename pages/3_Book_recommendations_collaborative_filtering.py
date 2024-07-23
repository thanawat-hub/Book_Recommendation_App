import streamlit as st

import numpy as np
from scipy.sparse.linalg import svds


@st.cache_data
def load_data4user_rating_df():
    # Merge ratings with users
    sample_ratings_df_dna = st.session_state['ratings_df'].sample(10000).dropna()

    sample_users_df_dna = st.session_state['users_df'].dropna()

    user_rating_df = sample_ratings_df_dna.merge(sample_users_df_dna, left_on='User-ID', right_on='User-ID')

    # Merge the resulting DataFrame with books
    book_user_rating = st.session_state['books_df'].merge(user_rating_df,  left_on='ISBN', right_on='ISBN')

    book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
    book_user_rating.reset_index(drop=True, inplace=True)

    # Create a dictionary for unique book identifiers
    d = {}
    for i, j in enumerate(book_user_rating.ISBN.unique()):
        d[j] = i
    book_user_rating['unique_id_book'] = book_user_rating['ISBN'].map(d)


    users_books_pivot_matrix_df = book_user_rating.pivot(index='User-ID',
                                                         columns='unique_id_book',
                                                         values='Book-Rating').fillna(0)
    # st.write(users_books_pivot_matrix_df.shape)#(3407, 5906)

    return book_user_rating, users_books_pivot_matrix_df

def get_book_id_from_name(book_user_rating, book_name):
    # Ensure the comparison is case-insensitive
    book_user_rating['Book-Title'] = book_user_rating['Book-Title'].str.lower()
    book_name = book_name.lower()
    try:
        book_id = book_user_rating[book_user_rating['Book-Title'] == book_name]['unique_id_book'].values[0]
        return book_id
    except IndexError:
        st.error(f"Book name '{book_name}' not found in the dataset.")
        return None

def top_cosine_similarity(data, book_id, top_n=10):
    index = book_id
    book_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(book_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def similar_books(book_user_rating, book_name, top_indexes):
    st.subheader('Recommendations for {0}: \n'.format(book_name))
    for id in top_indexes + 1:
        st.write(book_user_rating[book_user_rating.unique_id_book == id]['Book-Title'].values[0])

def book_recsys_collaborative_filtering():
    users_books_pivot_matrix_df = st.session_state['users_books_pivot_matrix_df'].values

    #>>>>>> # Performs matrix factorization of the original user item matrix # <<<<<<#
    # U, sigma, Vt = svds(users_books_pivot_matrix_df, k=NUMBER_OF_FACTORS_MF)
    # # U.dot(np.diag(sigma).dot(Vt))

    # # Reconstruct the ratings matrix
    # reconstructed_matrix = np.dot(U, np.dot(np.diag(sigma), Vt))
    #
    # # Visualize the original and reconstructed matrices
    # st.subheader("Original Ratings Matrix")
    # st.write(pd.DataFrame(users_books_pivot_matrix_df))
    #
    # st.subheader("Reconstructed Ratings Matrix")
    # st.write(pd.DataFrame(reconstructed_matrix))
    #
    # st.subheader("User Embeddings (U)")
    # st.write(pd.DataFrame(U))
    #
    # st.subheader("Item Embeddings (Vt)")
    # st.write(pd.DataFrame(Vt))
    #
    # # Plot the singular values
    # fig, ax = plt.subplots()
    # ax.plot(sigma, 'o-')
    # ax.set_title('Singular Values')
    # ax.set_xlabel('Index')
    # ax.set_ylabel('Value')
    # st.pyplot(fig)
    #
    # # Book recommendation
    # book_name = st.text_input("Enter a book name for recommendations")
    # if book_name:
    #     book_id = get_book_id_from_name(st.session_state['book_user_rating'], book_name)
    #     if book_id is not None:
    #         top_indexes = top_cosine_similarity(reconstructed_matrix, book_id)
    #         similar_books(st.session_state['book_user_rating'], book_name, top_indexes)

    # เราใช้ SVD เพื่อหาค่าแฟกเตอร์ที่ซ่อนอยู่ (Latent Factors) ที่แทนความสัมพันธ์ระหว่างผู้ใช้และรายการ

    NUMBER_OF_FACTORS_MF = 10
    st.write("number of factors mf ถ้า แฟกเตอร์ที่ซ่อนอยู่ มีจำนวนมาก -> ส่งผลให้ ระบบสามารถจับความสัมพันธ์ระหว่างผู้ใช้และรายการได้ละเอียดมากขึ้น = นานขึ้น")
    # ขั้นตอนนี้เป็นการแบ่งแยกข้อมูลผู้ใช้และหนังสือออกเป็นปัจจัยต่าง ๆ (เหมือนการแยกข้อมูลผู้ใช้และหนังสือออกเป็นกลุ่ม ๆ ตามปัจจัยที่เรากำหนด)
    U, sigma, Vt = svds(users_books_pivot_matrix_df, k=NUMBER_OF_FACTORS_MF)
    #Vt คือเมทริกซ์ที่ได้จากการทำ Singular Value Decomposition (SVD) ที่ประกอบด้วยข้อมูลเกี่ยวกับหนังสือทั้งหมด
    sigma = np.diag(sigma)
    #ขั้นตอนนี้เป็นการแบ่งแยกข้อมูลผู้ใช้และหนังสือออกเป็นปัจจัยต่าง ๆ (เหมือนการแยกข้อมูลผู้ใช้และหนังสือออกเป็นกลุ่ม ๆ ตามปัจจัยที่เรากำหนด)

    st.subheader("All user predicted ratings (คาดการณ์ว่าผู้ใช้แต่ละคนจะให้คะแนนหนังสือแต่ละเล่มอย่างไร)")
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    st.dataframe(all_user_predicted_ratings)
    # must show but long process time

    # Inputs for k and top_n
    k = st.number_input("Enter the number of components (k) เปรียบเทียบว่าเรากำหนดจำนวนปัจจัยที่เราต้องการวิเคราะห์ เช่น กำหนดปัจจัยในการประเมินหนังสือ k อย่าง เช่น ผู้แต่ง เรื่องราว การวิจารณ์ เป็นต้น", min_value=1, value=50)
    top_n = st.number_input("Enter the number of recommendations (top_n)", min_value=1, value=3)

    #ขั้นตอนนี้คือการสุ่มชื่อหนังสือ 3 เล่มจาก dataset (เปรียบเทียบกับการเลือกหนังสือสุ่มจากห้องสมุด)
    book_titles = st.session_state['book_user_rating']['Book-Title'].unique()
    # Set the seed for reproducibility
    input_number = st.number_input("Insert a number to random name of book in datasets", min_value=1)
    np.random.seed(int(input_number))
    random_elements = np.random.choice(book_titles, size=3)
    options = [f"{title}" for title in random_elements] + ["Another_option"]
    selection = st.selectbox("Select Book name list", options=options)
    #ขั้นตอนนี้คือการสุ่มชื่อหนังสือ 3 เล่มจาก dataset (เปรียบเทียบกับการเลือกหนังสือสุ่มจากห้องสมุด)

    # Create text input for user entry
    if selection == "Another_option":
        otherOption = st.text_input("Enter your other Book...")

    # Example usage
    sliced = Vt.T[:, :k]  # representative data
    #บรรทัดนี้หมายถึงการเลือกข้อมูลบางส่วนจากเมทริกซ์ Vt โดยเลือกคอลัมน์ตั้งแต่คอลัมน์ที่ 1 จนถึงคอลัมน์ที่ k (จำนวนที่ผู้ใช้กำหนด) แล้วทำการทรานสโพส (Transpose) เมทริกซ์ Vt
    # Vt.T[:, :]ในชีวิตจริง ถ้าเราเปรียบเทียบข้อมูลทั้งหมดของหนังสือในห้องสมุด (หรือในระบบแนะนำหนังสือ) การเลือกข้อมูลบางส่วนจาก Vt.T[:,] เปรียบเทียบกับการเลือกเฉพาะปัจจัยบางส่วนที่เราสนใจในการวิเคราะห์หนังสือ เช่น การเลือกพิจารณาเฉพาะข้อมูลเกี่ยวกับการรีวิว การให้คะแนน และความนิยมของหนังสือ จากปัจจัยทั้งหมดที่มีในระบบ
    # การเลือกเฉพาะ k ปัจจัยในการวิเคราะห์ เปรียบเทียบกับการเลือกดูข้อมูลเฉพาะบางส่วนที่สำคัญและมีผลต่อการแนะนำหนังสือ เช่น เลือกดูเฉพาะปัจจัยที่เกี่ยวข้องกับความชอบของผู้ใช้ในประเภทของหนังสือ หรือการรีวิวจากผู้อ่าน
    # สรุป การใช้ sliced = Vt.T[:, :k] ในการแนะนำหนังสือ คือการเลือกและใช้เฉพาะข้อมูลที่เป็นตัวแทนจากเมทริกซ์ Vt ซึ่งมีความสำคัญในการวิเคราะห์และแนะนำหนังสือ โดยเลือกเฉพาะปัจจัย k ที่เราสนใจและมีผลต่อการแนะนำมากที่สุด ซึ่งในชีวิตจริงเปรียบเทียบกับการเลือกดูข้อมูลเฉพาะบางส่วนที่สำคัญสำหรับการแนะนำหนังสือให้ผู้ใช้
    selected_book = selection if selection != "Another_option" else otherOption

    if selected_book:
        book_id = get_book_id_from_name(st.session_state['book_user_rating'], selected_book)
        if book_id is not None:
            similar_books(st.session_state['book_user_rating'], selected_book, top_cosine_similarity(sliced, book_id, top_n))


if __name__ == '__main__':

    # the_key = None
    # for the_key in st.session_state.keys():
    #     if the_key == "books_df4content_base":
    #         del the_key
    #     else:
    #         pass

    st.title('Book recommended system')
    st.image("D:/________Z____________Desktop_now/Pim-non/recommanded_system/NFC_book/datasets/images/collaborative_filtering_half.png", caption='Neural Collaborative Filtering')

    if 'users_books_pivot_matrix_df' not in st.session_state:
        st.session_state['book_user_rating'], st.session_state['users_books_pivot_matrix_df'] = load_data4user_rating_df()

    if 'users_books_pivot_matrix_df' and 'book_user_rating' in st.session_state:
        st.subheader("Dataframe preprocess from original data to use")
        st.subheader("book_user_rating head df")
        st.dataframe(st.session_state['book_user_rating'].head())

        st.subheader("then pivot columns unique_id_book and index is User-ID using values = Book-Rating (fill NA with 0)")
        st.write("note: first column is User-ID and row of top column is unique_id_book")
        st.dataframe(st.session_state['users_books_pivot_matrix_df'].head())# show ได้แค่อะไรที่ไม่เกิน 200MB
        # st.session_state['users_books_pivot_matrix_df'].shape#(3407, 5906)

        # error on streamlit -> MessageSizeError: Data of size 1891.3 MB exceeds the message size limit of 200.0 MB.
        book_recsys_collaborative_filtering()
    else:
        st.warning("Go to page main first")

