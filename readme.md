# Semantic QbE Evaluation on the Flickr Audio Captions Corpus

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](license.md)


## Overview

This code performs the evaluation for the semantic query-by-example (QbE)
speech search task described in the paper:

> H. Kamper, A. Anastassiou, and K. Livescu, "Semantic query-by-example speech
> search using visual grounding," in *Proc. ICASSP*, 2019.
> [[arXiv](https://arxiv.org/abs/1904.07078)]

Please cite this paper if you use the code.

The code here only performs the evaluation, i.e. model training is not
performed here. The models used in the above paper are very similar to those
produced with [this
recipe](https://github.com/kamperh/recipe_semantic_flickraudio).

The Flickr Audio Captions Corpus is available
[here](https://groups.csail.mit.edu/sls/downloads/flickraudio/).


## Task and data

Given a collection of utterances, the goal is to find which utterances contain
a given search query.

The queries are given in the `data/{dev,test}_query_segments.txt` files. Each
line is

    query_key word_type utterance start_time duration

The search utterances are given in `data/{dev,test}_search_segments.txt` which
simply lists the search utterances.

Semantic QbE evaluation is only possible with the test set, so all tuning
should first performed using exact QbE evaluation on the development data.


## Usage

To evaluate a model you need to produce a JSON file of model costs. E.g. the
JSON dictionary might look as follows:

    {
        "black_004_2196107384_361d73a170_0_000114-000153": {
            "017_3139876823_859c7d7c23_3": 0.39815978318610645, 
            "020_2393264648_a280744f97_4": 0.34421340893642866, 
            ...
            "033_2698666984_13e17236ae_1": 0.40632184740265026
        }, 
        "ball_005_2813033949_e19fa08805_1_000277-000322": {
            "017_3139876823_859c7d7c23_3": 0.3923965011047507, 
            "020_2393264648_a280744f97_4": 0.3770868000815181,
            ...
            "033_2698666984_13e17236ae_1": 0.31047880965214886
        }, 
        "carrying_007_3498327617_d2e3db3ee3_4_000132-000179": {
            "017_3139876823_859c7d7c23_3": 0.38565353872676705, 
            "020_2393264648_a280744f97_4": 0.4341039992463318,
            ...
        },
        ...
    }

For each query ID and for each utterance this cost dictionary should give the
cost for whether the query occurs in the utterance. Lower means that the model
believes the query occurs in that utterance. E.g. if the value of

    cost_dict["black_004_2196107384_361d73a170_0_000114-000153"]["017_3139876823_859c7d7c23_3"]

is very small, then this indicates that the model believes this query (an
instance of the word "black") occurs in utterance
`017_3139876823_859c7d7c23_3`.

After creating the JSON for all queries and search utterances, evaluation can
be performed as follows:

    ./eval_semantic_qbe.py cost_dict.json


## Example

To follow this example, download
<https://github.com/kamperh/flickr_semantic_qbe_eval/releases/download/v1.0/dtw_costs_test.json.zip>
and extract it in `exp/`. These are the baseline DTW costs.

Using the example file stored in `exp/dtw_costs_test.json`, if you run

    ./eval_semantic_qbe.py exp/dtw_costs_test.json

you should get the scores

    --------------------------------------------------------------------------
    Exact QbE:
    EER:  0.3213, avg: 0.3169, median: 0.3167, max: 0.4220, min: 0.2203
    P@10: 0.5456, avg: 0.4416, median: 0.4500, max: 0.7537, min: 0.0821
    P@N:  0.2487, avg: 0.2106, median: 0.2093, max: 0.3485, min: 0.0790
    --------------------------------------------------------------------------
    Semantic QbE:
    EER:  0.3865, avg: 0.3939, median: 0.3908, max: 0.4856, min: 0.3119
    P@10: 0.4428, avg: 0.3355, median: 0.3410, max: 0.5970, min: 0.0806
    P@N:  0.2430, avg: 0.1991, median: 0.1989, max: 0.3271, min: 0.0761
    Spearman's rho: 0.1368
    --------------------------------------------------------------------------

which matches the DTW baseline results on the test set in [the
paper](https://arxiv.org/abs/1904.07078).


