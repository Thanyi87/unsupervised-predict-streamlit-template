"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Introduction"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
     
        if page_selection == "Introduction":
            st.title("Welcome to saving time on binge watching")
            st.write("In today’s technology driven world, recommender systems are socially and economically critical for ensuring that                           individuals can make appropriate choices surrounding the content they engage with on a daily basis. One                                     application where this is especially true surrounds movie content recommendations; where intelligent algorithms                             can help viewers find great titles from thousands of options.With this context, EDSA is challenging us to                                   construct a recommendation algorithm based on content or collaborative filtering, capable of accurately monitor                             how a user will rate a movie they have not yet viewed based on their historical preferences.Providing an accurate                           and robust solution to this challenge has immense economic potential, with users of the system being exposed to                             content they would like to view or purchase - generating revenue and platform affinity.")
            
        st.subheader("Collaborators:")
        st.write("Sihle Riti")
        st.write("Nomfundo Manyisa")
        st.write(" Kwanda Silekwa")
        st.write("Thanyani Khedzi")
        st.write("Thembinkosi Malefo")
        st.write("Ofentse Makeketlane")
        st.subheader("sources")
        
        st.write("This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation                    service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of                        explicitly-based recommender systems, and now you get to as well! We'll be using a special version of the MovieLens                          dataset which has enriched with additional data, and resampled for fair evaluation purposes.The data for the MovieLens                      dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the                          University of Minnesota. Additional movie content data was legally scraped from IMDB")
    
    # ------------- SAFE FOR ALTERING/solution overview -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.subheader(" Collaborative modelling")
        st.write("We are going to give a brief explaination of collaborative filtering. Collaborative filtering is a technique that can                       filter out items that a user might like on the basis of reactions by similar users.It works by searching a large group of                   people and finding a smaller set of users with tastes similar to a particular user. It looks at the items they like and                     combines them to create a ranked list of suggestions to be more precise it is based on similarity in preference , taste                     and choices of two users. A good example that we can give you could be if user A, likes movies 1,2 and 3 and user B, likes                       movies 2,3 and 4 then this implies that they have similar interests and user A should like movie 4 and B should like                         movie 1.")
        st.subheader(" Content_based modelling")
        st.write("This filtering is based on the description or some data provided for that product. The system finds the similarity                         between recommended items based on their description or context. The user’s historical preference is taken into account                     to find products they may like in the future. For instance, if a user likes movies such as ‘Man in black’ then we can                       recommend him the movies of ‘Will Smith’ or movies with the genre ‘Sci-fi’.")
        st.subheader("Content based modelling vs Collaborative modelling???")
        st.write("Collaborative filtering recommender engine is a much better algorithim then content content based filtering since it is                     able to do feature learning on its own, in other words it can learn which features to use.")
      

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.


if __name__ == '__main__':
    main()
