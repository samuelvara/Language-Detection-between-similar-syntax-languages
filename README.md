# Language-Detection-between-similar-syntax-languages
All my project files
This project was done as part of my internship at the National University of Singapore.

The first phase of this project dealt with gather cleaned text dataset which I did by scraping data from 500,000 conversations from Reddit.

All the romance languages were considered for this project.
Since the romance languages share the same word dictionary but differ in their sentence formation, traditional machine learning algorithms would not work.
Ranking and predicting the language of a sentence also doesn't work since at an average the sentences share the same frequency of word (which are shared).

For the Natural Language Processing (NLP) model, I used Long Short Term Memory networks and Recurrent Neural Networks as the previous seen segment is important to accurately model the dataset.


The Neural Network model I designed works flawlessly when the length of sentence is more than 10 words.

Additional Optimizations to be done:
-Tune the model to work on shorter sentences
-Model should be trained with wider dialects
