**ISMT S-117: Text Analytics & NLP Final Project**
==================================================

Title: Fine-tuning BERT for News Category Classification
--------------------------------------------------------

### Project Members:

1.  Praneet Singh Solanki
    > ([*prs184@g.harvard.com*](mailto:prs184@g.harvard.com))

2.  Emmanuel Awa ([*ema142@g.harvard.com*](mailto:ema142@g.harvard.com))

### Overview

-   **Research question -** Enable users quickly classify news articles
    > into different categories of interest.

-   **What problem it solves -** With the influx of continuous
    > information available, is it possible to build a system that can
    > enable users to filter out different types of news content? This
    > information can be further used by a recommendation engine to
    > personalize news delivery of relevant categories to the users.

-   **What are different challenges -** Below are some challenges

    -   There is a lot of unstructured data. Most news articles do not
        > follow a single pattern which would make it easy to curate.

    -   The dataset we will be using has the news label, title and url
        > of the actual news, but does not contain the actual news
        > content. We would need to leverage BeautifulSoup and requests
        > libraries, respectively, to scrap the content. We will need to
        > implement an intelligent approach to the scaping such as
        > backoff algorithm, exponential request waiting to avoid
        > request throttling. Also we will ensure adherence to the
        > policy of the web page we’ll be scraping from.

### Inspiration

Our team at Microsoft focuses on making AI real through research and
engagements with customers. Our charter, in the next few months, is
focusing on building out a complete set of NLP toolkit with reusable
components that we can use while engaging in NLP projects. Based on past
engagements and learnings, one of our focus is on text classification of
news articles. This will feed into a bigger pipeline of capabilities
that will be operationalized for our daily work.

As members of the team, we have been tasked to find a way to bridge the
gap between this final project and our daily job in a way of creating
some of these reusable components. We’ve choose to do the news
classification for the following reasons:

-   **Availability of data** - Microsoft News team realized a new
    > challenge called
    > [*MIND*](https://msnews.github.io/index.html#getting-start)
    > consisting of a large corpora of news articles with labeled
    > categories as part of a recommendation challenge. We believe this
    > dataset can be leveraged for supervised NLP as well.

-   **Applicability of skill set to current day job at Microsoft -** As
    > mentioned above, we believe this final project will help us build
    > up skills for our actual day to day work. The learnings will
    > definitely be directly applicable.

-   **Research** - Text classification is a good place to start building
    > up NLP project portfolios

-   **Relevance -** With the current climate of fake news, having a
    > system that can distinguish what a news article is can feed into a
    > bigger system that can further classify if the news is real or
    > fake

### Dataset

The MIND dataset is publicly available for research. We plan to take the
news.tsv dataset (both train/val) for our news classification. It has 7
columns, which are divided by the tab symbol:

-   News ID

-   Category

-   SubCategory

-   Title

-   Abstract

-   URL

-   Title Entities (entities contained in the title of this news)

-   Abstract Entities (entities contained in the abstract of this news)

An example is shown in the following table:

  Column             Content
  ------------------ --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  News ID            N37378
  Category           sports
  SubCategory        golf
  Title              PGA Tour winners
  Abstract           A gallery of recent winners on the PGA Tour.
  URL                [*https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata*](https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata)
  Title Entities     \[{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": \[0\], "SurfaceForms": \["PGA Tour"\]}\]
  Abstract Entites   \[{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": \[35\], "SurfaceForms": \["PGA Tour"\]}\]

The most important columns are **Category, SubCategory, Title, Abstract
and URL**. The actual news content is not included in the dataset, but
it is left to the user to scrape it from the provided URL. We will be
using **web crawler/beautiful soup to get the actual news content**. As
the text size is going to be huge, we plan to use GPU virtual machines
on Azure for data prep and model training.

How does it helps to solve the problem:

This is a huge labeled dataset. It will definitely help us to build a
strong news classification model, which then can be used to classify the
unlabeled news articles. The labeled articles then can be used by a
recommendation engine to recommend news to users of the categories they
love.

### Exploratory Data Analysis

We plan to use the Title and Abstract of the article to predict the
category of that article. Here are the proposed EDA steps:

-   Clean up data and filter to use required features

-   Check the division of dataset by categories

-   Use tokenizer to remove unnecessary words/characters

-   Transforms the tokens to vectors using CV & TFIDF

-   Transforms the vectors to Topic model

-   Do the analysis of to 10 words for each news category

-   Determine the cosine similarity between the same news categories and
    > between diff news categories

### Methodology

Text classification is a supervised learning method of learning and
predicting the category or the class of a document given its text
content. With this we will need to perform the following steps

1.  **Document ingestion and preprocessing -** The MIND challenge
    > provides the dataset with the URL of the actual news articles. We
    > would need to build out a robust ingestion pipeline to fetch all
    > the news content, preprocess and save to flat files. Python
    > requests package and BeautifulSoup will be used here as it

2.  **Tokenization Strategy -** We will be employing the tokenization
    > strategy we learned in class; removing stop words, punctuations,
    > and urls. Spacy library will be used here.

3.  **Learning embeddings -** We will be learning the embeddings of the
    > news text using DistilBERT from hugging face library. We chose
    > distillbert because it’s a state-of-the-art transformer model that
    > is knowledge distilled from BERT, achieving similar accuracies
    > while having much lesser parameters for training and inference.

4.  **Model training -** The learned BERT embeddings will be fed into a
    > pre trained DistillBERT uncased architecture from Hugging Face
    > library

5.  **Fine tuning -** As these models are trained on very large
    > datasets, which includes multilingual corpora, we will need to
    > fine-tune our classifier specifically for the news article
    > category task. When fine-tuning for any task, additional data, not
    > used in the pre-trained model, is used to change the weights on
    > lower levels so that your model is better prepared for the context
    > of news category prediction.

6.  **Model Evaluation -** Our evaluation metrics for this final project
    > would be the F1 scores of the classified categories.

### Deployment Strategy

We plan to use Azure for this project. **This is an extended scope and
we will be doing only if we have more time**. The ideas is to use the
Azure Machine Services for following:

-   Train the NLP model on a GPU cluster and track its performance

-   Deploy the model as a web service on Azure Kubernetes Services for
    > inference

-   Build MLOps pipeline to automate the model training and model
    > inference

The deployed model will expose an API endpoint, which when called with
input parameters, will provide a response indicating the category of the
news article.
