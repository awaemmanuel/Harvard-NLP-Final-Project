**ISMT S-117: Text Analytics & NLP Final Project**
==================================================

Title: Fine-tuning BERT for News Category Classification
--------------------------------------------------------

### Project Members:

1.  Praneet Singh Solanki ([*prs184@g.harvard.com*](mailto:prs184@g.harvard.com))
2.  Emmanuel Awa ([*ema142@g.harvard.com*](mailto:ema142@g.harvard.com))

## Overview

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

## Dataset 

The MIND dataset is publicly available for research. We plan to take the [news.tsv](https://msnews.github.io/#getting-start) dataset (both train/val) for our news classification. It has 7 columns, which are divided by the tab symbol:

* News ID 
* Category 
* SubCategory
* Title
* Abstract
* URL
* Title Entities (entities contained in the title of this news)
* Abstract Entities (entites contained in the abstract of this news)

An example is shown in the following table:

Column | Content
------------- | -------------
News ID | N37378
Category | sports
SubCategory | golf
Title | PGA Tour winners
Abstract | A gallery of recent winners on the PGA Tour.
URL | https://www.msn.com/en-us/sports/golf/pga-tour-winners/ss-AAjnQjj?ocid=chopendata
Title Entities | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [0], "SurfaceForms": ["PGA Tour"]}]	
Abstract Entites | [{"Label": "PGA Tour", "Type": "O", "WikidataId": "Q910409", "Confidence": 1.0, "OccurrenceOffsets": [35], "SurfaceForms": ["PGA Tour"]}]

The most important columns are Category, SubCategory, Title, Abstract  and URL. The actual news content is not included in the dataset, but
it is left to the user to scrape it from the provided URL. We have used the columns **Category, Title, Abstract**, where we combined the Title and Abstract to form text which is then used to predict the Category. We did so as Title and Abstract consists of most important info/summary of the article, and must be enough to predict the category of article. Also, the text size from actual articles might be too big and would require distributed model training with cluster of GPU computers, which is very expensive. 

**How does it helps to solve the problem:**
> This is a huge labeled dataset. It will definitely help us to build a strong news classification model, which then can be used to classify the
unlabeled news articles. The labeled articles then can be used by a recommendation engine to recommend news to users of the categories they love.

## Exploratory Data Analysis

As mentioned in dataset description, we have used the Title and Abstract of the article to predict the category of that article. Following are the exploratory data analysis we did:

1. Clean up data and filter to use required features

1. Check the division of dataset by categories

![](assets/images/article_categories.png)

1. Use tokenizer to remove unnecessary words/characters

1. Transforms the tokens to vectors using CV & TFIDF

1. Transforms the vectors to Topic model NMF/LDA

1. Analysis of to 10 words for each news category

    - Top 10 words for each category using CV and TFIDF Vecs

    ![](assets/images/vecs_top_10.png)

    - Top 10 words for each topic usinNMF and LDA Vecs

    ![](assets/images/topic_top_n.png)


1. PCA Analysis on CountVectorizer, TF-IDF, LDA, NMF and Glove vectors

![](assets/images/PCA_Vecs.png)


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

## Deployment Strategy

Deployment of this text classification end to end pipeline on Azure was an extended score due to the limited time. However, we prepared the complete deployment stratergy to train and operationalize the text classification model on Azure. 

### Architecture Diagram
![](assets/images/architecture.png)

### Architecture Flow

#### Data Ingestion

In this architecture we consider the data (news articles dataset) to be available on Azure blob storage. It is uploaded as a batch process to the blob storage once a day. The data is consumed in the pipeline, where its cleaned and transformed to prepare it as model training dataset

#### Train/Retrain Pipeline
 
ML train pipeline orchestrates the process of retraining the model in an asynchronous manner. Retraining can be triggered on a schedule or when new data becomes available by calling the published pipeline REST endpoint from previous step. This pipeline covers the following steps:

* **Train model**. The text classification model training python script is executed on the Azure Machine Learning **GPU** Compute resource, which generates a new trained model pkl file. Azure ML Compute can be scaled to a cluster of GPU nodes enabling parallel model training on multiple nodes to improve speed/performance.

* **Evaluate model**. This step is a an evaluation test to compares the new model with the existing model. Only when the new model performs better than old/existing mode, based on various matrics, then it get promoted. Otherwise, the model is not registered and the pipeline is canceled.

* **Register model**. The better retrained model is registered with the Azure ML Model registry. This service provides version control for the models along with metadata tags so they can be easily reproduced.


#### Model O16n/Deployment pipeline

This pipeline shows how to operationalize(o16n) the trained model and promote it safely across different environments. This pipeline is subdivided into two environments, QA and production:

#### QA environment

- **Model Artifact trigger.** Deployment pipelines get triggered every time a new artifact is available. A new model registered to Azure Machine Learning Model Management is treated as a release artifact. In this case, a pipeline is triggered for each new model is registered.

- **Package Model as Container.** The registered model is packaged together with scoring/inference script and Python dependencies (Conda YAML file) into an operationalization Docker image. The image automatically gets versioned through Azure Container Registry.

- **Deploy on Container Instances.** This service is used to create a non-production environment. The scoring/inference image is also deployed here, and this is mostly used for testing. Container Instances provides an easy and quick way to test the Docker image.

- **Test web service.** A simple API test makes sure the image container is successfully deployed.

#### Production environment

- **Deploy on Azure Kubernetes Service.** This service is used for deploying scoring/inference image as a web service at scale in a production environment.

- **Test web service.** A simple API test makes sure the image container is successfully deployed.

**The deployed model will expose an API endpoint, which when called with input parameters, will provide a response indicating the category of the news article.**
