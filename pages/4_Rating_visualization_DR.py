import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA

from sklearn.manifold import MDS, Isomap, TSNE

def dimensional_reduction():
    st.subheader("Please select method to visualize data (one-by-one)")
    col1, col2 = st.columns(2)
    with col1:
        PCA_selected = st.checkbox(":rainbow[PCA [decomposition]]")
    with col2:
        TSNE_selected = st.checkbox(":rainbow[TSNE [manifold]]")


    if PCA_selected:
        st.subheader('Your selected PCA')
        d = st.select_slider("Select PCA dimension",
                             options=[1, 2, 3])

        # #PCA
        if d == 3:
            X = PCA(n_components=3).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']]) # ouput เป็น numpy array
            X = pd.DataFrame(X, columns=['x', 'y', 'z']) # ทำกลับมาให้เป็น df # ตรงนี้คือการเปลี่ยน แกนเป็นแกนใหม่แล้ว ซึ่งจะชื่อ pc1 pc2 แต่ใช้ x,y,zให้เข้าใจตรงกัน ซึ่งไม่รู้นะว่าคืออะไร จาก 4 มิติ ที่มัน compress มาเป็นสิ่งใหม่ใน 3 มิติ ที่ rotate แกนไปแล้ว (= space ใหม่ที่รู้แค่ data point ใกล้กันไหม)
            X['Book-Rating'] = ratings_df_number['Book-Rating'] # index 4 ไปใส่ใน X ตัวแปรใหม่ที่ลด dim แล้ว
            fig = px.scatter_3d(X,
                                x='x',
                                y='y',
                                z='z',
                                color='Book-Rating')
            st.plotly_chart(fig)

        if d == 2:
            X = PCA(n_components=2).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
            X = pd.DataFrame(X, columns=['x', 'y'])
            X['Book-Rating'] = ratings_df_number['Book-Rating'] # เอามิติที่ 4 ไปใส่ใน X ตัวแปรใหม่ที่ลด dim แล้ว
            fig = px.scatter(X,    #
                            x='x',
                            y='y',
                            color='Book-Rating')
            st.plotly_chart(fig)
        if d == 1:
            X = PCA(n_components=1).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
            X = pd.DataFrame(X, columns=['x'])
            X['Book-Rating'] = ratings_df_number['Book-Rating'] # เอามิติที่ 4 ไปใส่ใน X ตัวแปรใหม่ที่ลด dim แล้ว
            X['y'] = 0
            fig = px.scatter(X, # แต่ engine นี้ รองรับ 2 มิติ เลย ลบ ไม่ได้ ก็กำหนด y ให้ = 0
                             x='x', # ตรงนี้คือการเรียกชื่อ columns
                             y='y', # ตรงนี้คือการเรียกชื่อ columns
                             color='Book-Rating')
            st.plotly_chart(fig)

    # if MDS_selected:
    #     st.subheader('MDS')
    #     d = st.select_slider("Select MDS dimension",
    #                          options=[1, 2, 3])
    #     if d == 3:
    #         X = MDS(n_components=3).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
    #         X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    #         X['Book-Rating'] = ratings_df_number['Book-Rating']
    #         fig = px.scatter_3d(X,
    #                       x='x',
    #                       y='y',
    #                       z='z',
    #                       color='Book-Rating')
    #         st.plotly_chart(fig)
    #     if d == 2:
    #         X = MDS(n_components=2).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
    #         X = pd.DataFrame(X, columns=['x', 'y'])
    #         X['Book-Rating'] = ratings_df_number['Book-Rating']
    #         fig = px.scatter(X,
    #                       x='x',
    #                       y='y',
    #                       color='Book-Rating')
    #         st.plotly_chart(fig)
    #     if d == 1:
    #         X = MDS(n_components=1).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
    #         X = pd.DataFrame(X, columns=['x'])
    #         X['Book-Rating'] = ratings_df_number['Book-Rating']
    #         X['y'] = 0
    #         fig = px.scatter(X,
    #                       x='x',
    #                       y='y',
    #                       color='Book-Rating')
    #         st.plotly_chart(fig)

    # # # Isomap
    # if Isomap_selected:
    #     # st.subheader('Isomap')
    #     # d = st.select_slider("Select Isomap dimension",
    #     #                      options=[1, 2, 3])
    #     # k = st.select_slider("Select Isomap neighbors",
    #     #                      options=list(range(3, 20)))
    #     # if d == 3:
    #     #     X = Isomap(n_components=3, n_neighbors=k).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
    #     #     X = pd.DataFrame(X, columns=['x', 'y', 'z'])
    #     #     X['Book-Rating'] = ratings_df_number['Book-Rating']
    #     #     fig = px.scatter_3d(X,
    #     #                   x='x',
    #     #                   y='y',
    #     #                   z='z',
    #     #                   color='Book-Rating')
    #     #     st.plotly_chart(fig)
    #     # if d == 2:
    #     #     X = Isomap(n_components=2, n_neighbors=k).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
    #     #     X = pd.DataFrame(X, columns=['x', 'y'])
    #     #     X['Book-Rating'] = ratings_df_number['Book-Rating']
    #     #     fig = px.scatter(X,
    #     #                   x='x',
    #     #                   y='y',
    #     #                   color='Book-Rating')
    #     #     st.plotly_chart(fig)
    #     # if d == 1:
    #     #     X = Isomap(n_components=1, n_neighbors=k).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
    #     #     X = pd.DataFrame(X, columns=['x'])
    #     #     X['Book-Rating'] = ratings_df_number['Book-Rating']
    #     #     X['y'] = 0
    #     #     fig = px.scatter(X,
    #     #                   x='x',
    #     #                   y='y',
    #     #                   color='Book-Rating')
    #     #     st.plotly_chart(fig)

    # # t-SNE
    if TSNE_selected:
        st.subheader('t-SNE')
        d = st.select_slider("Select t-SNE dimension",
                             options=[1, 2, 3])
        if d == 3:
            X = TSNE(n_components=3).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
            X = pd.DataFrame(X, columns=['x', 'y', 'z'])
            X['Book-Rating'] = ratings_df_number['Book-Rating']
            fig = px.scatter_3d(X,
                          x='x',
                          y='y',
                          z='z',
                          color='Book-Rating')
            st.plotly_chart(fig)
        if d == 2:
            X = TSNE(n_components=2).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
            X = pd.DataFrame(X, columns=['x', 'y'])
            X['Book-Rating'] = ratings_df_number['Book-Rating']
            fig = px.scatter(X,
                          x='x',
                          y='y',
                          color='Book-Rating')
            st.plotly_chart(fig)
        if d == 1:
            X = TSNE(n_components=1).fit_transform(ratings_df_number.loc[:, ['User-ID', 'user_id', 'item_id']])
            X = pd.DataFrame(X, columns=['x'])
            X['Book-Rating'] = ratings_df_number['Book-Rating']
            X['y'] = 0
            fig = px.scatter(X,
                          x='x',
                          y='y',
                          color='Book-Rating')
            st.plotly_chart(fig)
if __name__ == '__main__':

    st.title("Rating Visualization with DR")
    if 'user_idx' in st.session_state:
        ratings_df_number = st.session_state['ratings_df'].drop(['ISBN', 'user_idx', 'item_idx'], axis=1).copy(deep=True)[
                            :(st.session_state['ratings_df'].shape[0]) // 10000]

        dimensional_reduction()
    else:
        st.warning("Go to page Book_recommanded_ncf first")
