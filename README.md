# Sentiment Analysis and Information Extraction on Tourist Reviews #

My project for the HCL AI Hackathon 2019, the project consists of the following -
* ## Web Scrapers ##
The data used in the project is textual data scraped from TripAdvisor and Google Maps. Both of them are built using Selenium and BeautifulSoup. Scrapers can be found in the respective folder.
* TripAdvisor Scraper
* Google Maps Scraper

* ## Model ##
* ### Classifier ###
All the data preprocessing and model training code is provided in the `imdb-fastai.ipynb` file. Training of classifier was done on a mixture of IMDB movie reviews and tourist scraped data over a fastai language model with a promising 93.9% accuracy with just ~2 hrs of training on Tesla P100 from Kaggle.

* ### Topic Model ###
The topic model used for information extraction using LDA can be found in `topic-model.py`. Alongwith that the script also has our custom API functions built to it to serve creative visualizations and insights about data alongwith the sentiment related to it.

* ### Inference ###
The final file `presentation-notebook.ipynb` gets the inputs as raw text or .csv and analyses data to give insights to data with prediction and visualizations.
