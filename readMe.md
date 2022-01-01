# Movie recommender

This program is based on my coursework during the course data.ml.360 Recommender Systems at Tampere University. 

## Functionality

The program recommends a set of movies to a group of users from the MovieLens dataset. The program was made with the MovieLens 100k dataset in mind,
though it should work with the full dataset as well. It also is capable of answering a set questions regarding the recommendation it makes:
- Why is one movie higher than another
- Why is a movie not recommended
- Why is a genre of movies not recommended

It is also capable of producing sequences of movie recommendations, which aim to balance the recommendations in such a way that everyone is happy at the end.
You can read more about the theory of this in the research article: https://homepages.tuni.fi/konstantinos.stefanidis/docs/sac20.pdf

## Dependancies

The program uses the following libraries. It was implented with the 3.9.7 version of python.
- Pandas
- Numpy
- Scipy

## To implement

- Ask the location of the MovieLens files as a commandline parameter
- Ask the currently hardcoded parameters from the user
- Allow the user to decide whether they want a sequence of recommendations or one set, and which questions they want answered
- Possibly allow asking the questions using movie names rather than id's

## Current status

Currently the program only makes a single recommendation for a group, and answers hardcoded questions about it, mostly to show that the program works.
The functions for sequential recommendations have been added to the code, but currently aren't used.

## Limitations

- Somewhat slow
- Can not answer why-not questions on a sequential recommendation
- The function that makes the recommendations for a group isn't necessarily very good