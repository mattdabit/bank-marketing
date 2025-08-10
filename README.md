# bank-marketing

A repository focused on assessing whether someone will accept a marketing campaign

## Table of Contents

- [Link to notebook](#link-to-notebook)
- [Local Installation](#local-installation)
- [About the Dataset](#about-the-dataset)
- [Understanding the business](#understanding-the-business)
- [Understanding the features](#understanding-the-features)
- [Understanding the data](#understanding-the-data)
    - [Imbalance in the data](#anomalies-in-the-data)
    - [Univariate and Multivariate Analysis](#univariate-and-multivariate-analysis)
        - [Age analysis](#age-analysis)
        - [Job analysis](#job-analysis)
        - [Education analysis](#education-analysis)
        - [Month analysis](#month-analysis)
        - [Duration analysis](#duration-analysis)
        - [Calls analysis](#calls-analysis)
        - [Calls X Duration analysis](#calls-x-duration-analysis)
        - [Days since last contact analysis](#days-since-last-contact-analysis)
        - [Correlation Matrix](#correlation-matrix)
        - [Consumer price index analysis](#consumer-price-index-analysis)
        - [Consumer confidence index analysis](#consumer-confidence-index-analysis)
        - [Previous outcome analysis](#previous-outcome-analysis)

## Link to notebook

[Primary Notebook](https://github.com/mattdabit/bank-marketing/blob/main/bank_marketing.ipynb)

## Local Installation

1. Clone the repository
2. Use a python environment manager. I prefer conda.
3. Create and activate conda environment
    ```
    conda env create -f environment.yml   
    conda activate bank
    ```

## About the Dataset

The dataset was procured from
the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The data is from a
Portugese bank. It contains a collection of marketing campaigns. The dataset is not missing any values and only contains
12 duplicates. I was quite inspired by
the [paper](https://github.com/mattdabit/bank-marketing/blob/main/CRISP-DM-BANK.pdf) written by Sérgio Moro and Raul M.
S. Laureano.
In my analysis I did find that some of their data findings do not match what I have in the dataset.

## Understanding the business

It comes as no surprise that large marketing campaigns have negative sentiment amongst the general populace. Think about
the last time you answered an unexpected phone call from an unknown number, if your experience is anything like my then,
it was either a scam caller, telemarketer or survey taker. I find myself hanging up quickly when it comes to these types
of calls, if I were to ever answer them. Every failed cold call costs the company commissioning the campaign time and
money. The bank partner commissioning this study is seeking to increase campaign success and reduce costs by focusing on
profiles that are more likely to accept their offerings. The bank partner would like a model that can better predict the
type of person that would accept offers from our partner bank.

## Understanding the Features

```
Input variables:
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
```

## Understanding the data

The first thing that jumps out is how imbalanced the dataset is.
This is to be expected considering that we are working
with telemarketing data.
I was also able to find some strong predictors for the accepting class.
In particular, the
following fields show strong promise: month, employment, number of contacts.
It is also important to note that the pdays
column using 999 to signify that fact the client was not priorly contacted.
The author of this dataset also recommends
avoiding the duration column as it highly affects the output.

### Imbalance in the data

<img src="images/acceptance_count.png"/>

Here you can see the lack of balance. As spoken about before, this is expected with telemarketing campaigns.

### Univariate and Multivariate Analysis

#### Age analysis

<img src="images/age_box.png"/>

In the above box plot we see that a majority of contacts are between the ages of 32 and 47.

<img src="images/age_acceptance_ratio.png"/>

Here we notice that the younger and older contacts are more likely to accept. We see some strange behavior going on when
we creep pass the upper fence of the box plot. It may be valuable to introduce a cutoff age to ensure our models do not
over
index on age.

#### Job analysis

<img src="images/job_acceptance_ratio.png"/>

Given what we learned about the correlation between age and acceptance rate it should come to no suprise that students
and
retirees are more likely to accept a campaign.

#### Education analysis

<img src="images/education_acceptance_ratio.png"/>

I wanted to see how strong an impact education would have on concerning the target.
The correlation does not seem present; however, those that are illiterate are more likely than other categories to
accept the campaign.

#### Month analysis

<img src="images/month_acceptance_ratio.png"/>

Timing seems to be an important factor in these campaigns.
In particular, March, December, September and October all had high
acceptance rates.
We may be able to leverage this as a business to reduce costs during slow months.
This somewhat differs from the paper
which found that June was a very successful month.

#### Duration analysis

<img src="images/duration_acceptance_ratio.png"/>

The duration of a call seems to be a strong predictor; the longer someone is on the phone, the more likely they will accept.
However, we see this note from the author of the dataset.
`Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.`
Given that we will not be using this value in our modeling, but it is still important to study it. 

#### Calls analysis

<img src="images/calls_acceptance_ratio.png"/>

The number of calls may have a negative correlation with campaign acceptance.
That is the more you call a prospect, the more likely they will say no.

#### Calls X Duration analysis

<img src="images/calls_vs_duration.png"/>

I was surprised by this finding. I would have expected that the more calls would lead to brief conversation if a person
 picked up, but there is barely a correlation between the values. 

#### Days since last contact analysis

<img src="images/days_since_last_contact_acceptance_ratio.png"/>

The histogram above indicates that the best time to follow up with a customer is within the week of the first call.
I would not be surprised
if the banking institution provided guidance to their employees
to call on the 3rd and 6th days after their first call.  

#### Correlation Matrix

<img src="images/correlation.png"/>

Social and economic attributes are highly correlated.
This fact may lend itself to PCA or early feature pruning.
The number of contacts and number of days
that passed by after the client was last
being negatively correlated is also not surprising.
We should be careful of multicollinearity with our dataset.
This won't be an issue for decision trees and SVM models, but it will be for logistic regression. 

#### Consumer price index analysis

<img src="images/consumer_price_index_acceptance_ratio.png"/>

The consumer price index looks static across the various levels.
Now it seems like the ratio of acceptance goes up at the 94.25 value, 
but without more data I would hesitate to make a judgment. 

#### Consumer confidence index analysis

<img src="images/consumer_confidence_index_acceptance_ratio.png"/>

Consumer confidence may affect acceptance. I am not entirely convinced considering that the acceptance count
seems stagnant for the 3 index ratings with the most calls.
It is promising to see the ratio of acceptance jump with the `-32.5` rating.
However, this could just be an outlier. 

#### Previous outcome analysis

<img src="images/poutcome_acceptance_ratio.png"/>

Retention is over 65% for customers that previously accepted a campaign.
It may be prudent to contact these types of customers to see why they chose our product. 
