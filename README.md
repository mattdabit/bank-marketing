# bank-marketing

A repository focused on assessing whether someone will accept a marketing campaign

## Table of Contents

- [Link to notebook](#link-to-notebook)
- [Local Installation](#local-installation)
- [About the Dataset](#about-the-dataset)
- [Understanding the business](#understanding-the-business)
- [Understanding the features](#understanding-the-features)
- [Understanding the data](#understanding-the-data)
  - 

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
12 duplicates.

## Understanding the business

It comes as no surprise that large marketing campaigns have negative sentiment amongst the general populace. Think about
the last time you answered an unexpected phone call from an unknown number, if your experience is anything like my then,
it was either a scam caller, telemarketer or survey taker. I find myself hanging up quickly when it comes to these types
of calls, if I were to ever answer them. Every failed cold call costs the company commissioning the campaign time and
money. The bank partner commissioning this study is seeking to increase campaign success and reduce costs by focusing on
profiles that are more likely to accept their offerings. The bank partner would like a model that can better predict the
type of person that would accept offers from our partner bank.

### Understanding the Features

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
- [Understanding the data](#understanding-the-data)
