import streamlit as st
import plotly.express as px# ได้ html
import matplotlib.pyplot as plt

import numpy as np
from geopy.geocoders import Nominatim
from time import sleep

import plotly.graph_objects as go
def geocode_locations(df):
    geolocator = Nominatim(user_agent="my_app")
    df['lat'] = None
    df['lon'] = None

    batch_size = 10  # Adjust batch size as needed
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        for idx, row in batch_df.iterrows():
            location = row['Location']
            try:
                loc = geolocator.geocode(location)
                if loc:
                    df.at[idx, 'lat'] = loc.latitude
                    df.at[idx, 'lon'] = loc.longitude
            except Exception as e:
                st.write(f"Error geocoding {location}: {e}")
        sleep(1)  # Add a delay to avoid overwhelming the service


# Function to plot density using Plotly map
def plot_density_map(df):
    fig = px.density_mapbox(df, lat='lat', lon='lon', z='Age', radius=10,
                            center=dict(lat=np.mean(df['lat']), lon=np.mean(df['lon'])),
                            zoom=5, mapbox_style="carto-positron",
                            title='')
    st.plotly_chart(fig)

if __name__ == '__main__':
    # ดูภาพรวม, เจาะทีละ feature
    st.title('Exploratory Data Analysis: EDA')

    if 'datasets' in st.session_state:#จะเริ่มทำ ก็ต่อเมื่อ load data แล้ว
        # books_df = st.session_state['books_df']#หยิบมาใส่ในตัวแปรdf
        # ratings_df = st.session_state['ratings_df']
        # users_df = st.session_state['users_df']
        n = 10
        option = st.selectbox(
            f"How would you like to EDA {n} row which datasets",
            ("users_df", "books_df", "ratings_df"))

        if option == "books_df":
            st.subheader(f"About {option} files:")
            st.write("Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset.")
            st.write("Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), **obtained from Amazon Web Services.**")
            st.write("**Note that in case of several authors, only the first is provided**. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.")

            st.dataframe(st.session_state['books_df'][:n])
            st.write(f"In {option} all have row={st.session_state['books_df'].shape[0]} column={st.session_state['books_df'].shape[1]}")
            st.write("-----------------")
            # genre = st.radio( # it not work
            #     "What's your prefer from this options",
            #     [":rainbow[Correlation of this csv]", ":blue[***Count unique books for each author***]"])

            # ใช้ ตัวนี้แทน `tmp_books_df`
            tmp_books_df = st.session_state['books_df'].copy()
            # Preprocess the text columns by converting them to lowercase
            tmp_books_df['Book-Title'] = st.session_state['books_df']['Book-Title'].str.lower()
            tmp_books_df['Book-Author'] = st.session_state['books_df']['Book-Author'].str.lower()

            # Create a DataFrame with unique book titles and their authors
            unique_books_df = tmp_books_df.drop_duplicates(subset=['Book-Title'])

            # Calculate the length of each ISBN
            st.session_state['books_df']['ISBN_Length'] = st.session_state['books_df']['ISBN'].astype(str).apply(len)

            # Count the frequency of each length
            isbn_length_counts = st.session_state['books_df']['ISBN_Length'].value_counts().sort_index()
            # Streamlit app
            st.subheader('Frequency of ISBN Lengths')
            # Plot using matplotlib and display in Streamlit
            fig, ax = plt.subplots(figsize=(10, 6))
            isbn_length_counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title('Frequency of ISBN Lengths')
            ax.set_xlabel('ISBN Length')
            ax.set_ylabel('Frequency')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=0)
            st.pyplot(fig)
            detail = st.checkbox("Show detail?")
            if detail:
                st.dataframe(isbn_length_counts)

            st.write('---')
            st.subheader("Please select one-by-one")
            col1, col2 = st.columns(2)
            with col1:
                Year = st.checkbox(":rainbow[Plot Pie Year-Of-Publication]")
            with col2:
                CQBPEA = st.checkbox(":blue[***Count unique books for each author***]")

            if CQBPEA:
                # numbers_sample_plot = 2000#238964
                numbers_sample_plot = int(st.number_input("Insert a number a sample for plot", value=1, placeholder="Type a number..."))
                max_plot = st.checkbox("Want to see max plot?")
                if max_plot:
                    numbers_sample_plot = 238964
                sampled_books_df = unique_books_df.sample(n=numbers_sample_plot, random_state=1)

                # Count the number of unique books for each author
                auth_counts = sampled_books_df['Book-Author'].value_counts().reset_index()
                auth_counts.columns = ['Author', 'Count']
                # Create a bar chart using Plotly
                # fig = px.bar(auth_counts, x='Author', y='Count', title='Number of Unique Books per Author', color_discrete_sequence=['#636EFA'])
                # Create a bar chart using Plotly with a color gradient based on count
                fig = px.bar(auth_counts, x='Author', y='Count', title=f'Number of Unique Books per Author (sample for plot {numbers_sample_plot}/{unique_books_df.shape[0]}).', color='Count',
                             color_continuous_scale='Viridis')

                # Update the layout of the chart
                fig.update_layout(
                    xaxis_title='Author',
                    yaxis_title='Count of Books',
                    title_font_size=20,
                    height=500,
                    width=800
                )

                # Display the chart in Streamlit
                st.plotly_chart(fig)

            if Year:
                col = "Year-Of-Publication"
                tmp = unique_books_df[col][:unique_books_df[col].shape[0]]

                st.write(f"Number of Year-Of-Publication in pie (all={unique_books_df[col].shape[0]}).")
                fig = px.pie(tmp, names=col)
                st.plotly_chart(fig)

        if option == "users_df":
            st.subheader(f"About {option} files:")
            st.write("Contains the users.")
            st.write("**Note that user IDs (User-ID) have been *anonymized* and map to integers**.")
            st.write("Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values.")

            st.dataframe(st.session_state[option][:n])
            st.write(f"In {option} all have row={st.session_state[option].shape[0]} column={st.session_state[option].shape[1]}")
            st.write("___")

            # Age Distribution
            st.subheader('Interactive Age Distribution of Users')
            # Drop rows with missing age values
            users_df = st.session_state[option].dropna(subset=['Age'])
            # Convert Age to integer
            users_df['Age'] = users_df['Age'].astype(int)
            # Calculate the 99th percentile of the Age
            age_99_percentile = users_df['Age'].quantile(0.99)#ถ้าไม่ทำจะมี 0-244

            # Filter out the top 1% of age outliers
            filtered_users_df = users_df[users_df['Age'] <= age_99_percentile]
            # Determine the range for the x-axis
            age_min = filtered_users_df['Age'].min()
            age_max = filtered_users_df['Age'].max()

            # # Plot the age distribution
            # fig = px.histogram(filtered_users_df, x='Age', nbins=30, title='drop nan and filter quantile 99')
            #
            # # Customize the plot with the specified x-axis range
            # fig.update_layout(
            #     xaxis_title='Age',
            #     yaxis_title='Count of Users',
            #     xaxis=dict(range=[age_min, age_max]),
            #     bargap=0.2
            # )
            # # Display the plot in Streamlit
            # st.plotly_chart(fig)

            hist_data = filtered_users_df['Age']
            hist, bin_edges = np.histogram(hist_data, bins=30)

            # Calculate the normalized counts for the gradient effect
            normalized_counts = (hist - hist.min()) / (hist.max() - hist.min())

            # Create the bar colors based on the normalized counts
            colors = px.colors.sequential.Viridis

            # Generate colors for each bar
            bar_colors = [colors[int(value * (len(colors) - 1))] for value in normalized_counts]

            # Create the figure
            fig_filtered = go.Figure(
                data=[go.Bar(x=bin_edges[:-1], y=hist,
                             marker=dict(color=normalized_counts, colorscale='Viridis', colorbar=dict(title='Count')))]
            )

            # Customize the plot with the specified x-axis range
            fig_filtered.update_layout(
                title='Drop NaN and Filter Quantile 99',
                xaxis_title='Age',
                yaxis_title='Count of Users',
                bargap=0.2,
                coloraxis_colorbar=dict(title='Count')#
            )
            # Display the plot in Streamlit
            st.plotly_chart(fig_filtered)

            # st.write("----")
            # # Display the plot in Streamlit
            # st.plotly_chart(fig_filtered)
            #
            # location_counts = users_df['Location'].value_counts().reset_index()
            # location_counts.columns = ['Location', 'Count']
            # fig_location = px.bar(location_counts.head(10), x='Location', y='Count',
            #                       title='Top 10 Locations by User Count')
            # st.plotly_chart(fig_location)

            # Assuming 'users_df' is already defined and contains the data
            location_counts = users_df['Location'].value_counts().reset_index()
            location_counts.columns = ['Location', 'Count']

            st.subheader("Top 10 Locations by User Count")
            # Create the bar chart with gradient colors based on the count
            fig_location = px.bar(
                location_counts.head(10),
                x='Location',
                y='Count',
                title='',
                color='Count',
                # color_continuous_scale='Viridis'  # You can choose different scales like 'Viridis', 'Cividis', etc.
            )
            # Display the plot in Streamlit
            st.plotly_chart(fig_location)

            # st.write("----")
            st.subheader('Age Density by Location[sample show top users_df 30 data]')
            # df_sample = st.session_state[option][:50]
            df_sample = filtered_users_df[:30]
            # Geocode locations if not already geocoded
            geocode_locations(df_sample)
            # Display interactive map with density visualization
            plot_density_map(df_sample)

        if option == "ratings_df":
            st.subheader(f"About {option} files:")
            st.write("Contains the book rating information.")
            st.write("Ratings (Book-Rating) are either ")
            st.write("**explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation)**, or ")
            st.write("***implicit, expressed by 0.***")
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(st.session_state[option][:n])
                st.write(f"In {option} all have row={st.session_state[option].shape[0]} column={st.session_state[option].shape[1]}")
            with col2:
                st.write("Describe Summary statistics of the ratings")
                st.dataframe(st.session_state[option].describe())  # Standard Deviation บอกถึงการกระจายตัวของข้อมูล ว่าแตกต่างจาก means ที่ 3.85 แปลว่า มีการกระจายตัวที่หลากหลาย และไม่ค่อยมีความใกล้เคียงกันมากนัก
            st.write("___")

            # Distribution of book ratings
            st.subheader('Distribution of Book Ratings')
            rating_counts = st.session_state[option]['Book-Rating'].value_counts().sort_index()
            fig_rating_distribution = px.bar(rating_counts,
                                             x=rating_counts.index,
                                             y=rating_counts.values,
                                                 labels={'x': 'Book Rating', 'y': 'Count'},
                                             title='')
            st.plotly_chart(fig_rating_distribution)

            # st.markdown("### Select column for plot correlation")
            # options_col = st.multiselect(
            #     "",
            #     [col for col in st.session_state[option].columns])
            # # อาจจะ design ว่า ถ้าเลือก column ไหน ก็เอามา plot corr แต่ทำได้แค่ numeric
            # corr = st.session_state[option][options_col].corr(numeric_only=True).round(2)#2ตำแหน่ง
            # fig = px.imshow(corr, text_auto=True)#plot heat map
            # # fig.show()#ถ้าใส่มันจะเด้งไป port ของ plotly
            # st.plotly_chart(fig)#ต้องเรียกใช้บน st เนอะ ว่าใช้ interactive ตัวไหน


    else:
        st.warning("Go to page main first")
    #--- for check ---#
        # for the_key in st.session_state.keys():
        #     st.write(the_key)
    #--- for check ---#

        # # อาจจะ design ว่า ถ้าเลือก column ไหน ก็เอามา plot corr แต่ทำได้แค่ numeric
        # corr = df[selected_col].corr(numeric_only=True).round(2)#2ตำแหน่ง
        # fig = px.imshow(corr, text_auto=True)#plot heat map
        # # fig.show()#ถ้าใส่มันจะเด้งไป port ของ plotly
        # st.plotly_chart(fig)#ต้องเรียกใช้บน st เนอะ ว่าใช้ interactive ตัวไหน
        #
        #
        # col = st.sidebar.selectbox('Select columns', df.columns)
        # tmp = df[col]
        # if pd.api.types.is_numeric_dtype(tmp):
        #     outliers = st.sidebar.checkbox('Outliers', False)#สำหรับตัวที่เป็นเลข ก็ควรมีการตัด outliers
        #     if outliers:
        #         q_low = tmp.quantile(0.01)
        #         q_high = tmp.quantile(0.99)
        #         tmp = tmp[(tmp > q_low) & (tmp < q_high)]
        #     st.write(tmp.describe())
        #     fig = px.histogram(tmp, x=col)
        #     st.plotly_chart(fig)
        #
        # else:#เพราะไม่ใช่ตัวเลข
        #     st.write(tmp.value_counts())
        #     fig = px.pie(tmp, names=col)
        #     st.plotly_chart(fig)
        #     #plot pie chart ว่ากินพื้นที่เท่าไหร่
        #
        #
        # # ถ้าค่ามันไปกระจุกอยู่ที่หนึ่งก็ให้ กำจัด outliner ออกไป
        # # ถ้าเกิน ช่วงนั้นตัดทิ้ง