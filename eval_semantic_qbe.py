#!/usr/bin/env python

"""
Evaluate semantic QbE performance for a given costs directory in JSON format.

Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2017, 2018, 2021
"""

from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.stats import spearmanr
from tqdm import tqdm
import argparse
import json
import numpy as np
import sklearn.metrics as metrics
import sys


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "json_fn", type=str, help="JSON file with the cost dictionary"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_captions_dict(captions_fn):
    captions_dict = {}
    with open(captions_fn) as f:
        for line in f:
            line = line.strip().split()
            captions_dict[line[0]] = [
                i for i in line[1:] if "<" not in i and not "'" in i
                ]
    return captions_dict


#-----------------------------------------------------------------------------#
#                             EVALUATION FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def calculate_eer(y_true, y_score):
    # https://yangcha.github.io/EER-ROC/
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer


def eval_qbe(cost_dict, label_dict, analyse=False):
    """
    Return dictionaries of P@10, P@N and EER for each query item.

    The keys of each of the returned dictionaries are the unique keyword types,
    with the value a list of the scores for each of the queries of that keyword
    type.
    """

    # Unique keywords with query keys
    keyword_dict = {}
    for query_key in cost_dict:
        keyword = query_key.split("_")[0]
        if keyword not in keyword_dict:
            keyword_dict[keyword] = []
        keyword_dict[keyword].append(query_key)

    # For each keywords
    eer_dict = {}  # `eer_dict[keyword]` is a list of EER scores for each query
                   # of that keyword type
    p_at_10_dict = {}
    p_at_n_dict = {}
    if analyse:
        print
    for keyword in tqdm(sorted(keyword_dict)):

        eer_dict[keyword] = []
        p_at_10_dict[keyword] = []
        p_at_n_dict[keyword] = []

        # For each query key
        for query_key in sorted(keyword_dict[keyword]):

            # Rank search keys
            utt_order = [
                utt_key for utt_key in sorted(cost_dict[query_key],
                key=cost_dict[query_key].get) if utt_key in label_dict
                ]

            # EER
            y_true = []
            for utt_key in utt_order:
                if keyword in label_dict[utt_key]:
                    y_true.append(1)
                else:
                    y_true.append(0)
            y_score = [cost_dict[query_key][utt_key] for utt_key in utt_order]

            cur_eer = calculate_eer(y_true, [-i for i in y_score])
            eer_dict[keyword].append(cur_eer)

            # P@10
            cur_p_at_10 = float(sum(y_true[:10]))/10.
            p_at_10_dict[keyword].append(cur_p_at_10)

            # P@N
            cur_p_at_n = float(sum(y_true[:sum(y_true)]))/sum(y_true)
            p_at_n_dict[keyword].append(cur_p_at_n)

            if analyse:
                print("-"*79)
                print("Query:", query_key)
                print(f"Current P@10: {cur_p_at_10:.4f}")
                print(f"Current P@N: {cur_p_at_n:.4f}")
                print(f"Current EER: {cur_eer:.4f}")
                # print("Top 10 utterances: ", utt_order[:10])
                print("Top 10 utterances:")
                for i_utt, utt in enumerate(utt_order[:10]):
                    print(
                        "{utt}: {}".format(utt, " ".join(label_dict[utt])),
                        end=""
                        )
                    if y_true[i_utt] == 0:
                        print(" *")
                    else:
                        print()

    if analyse:
        print("-"*79)
        print()

    return eer_dict, p_at_10_dict, p_at_n_dict


def get_avg_scores(score_dict):
    """
    Return the overall average, and unweighted average, median and maximum
    scores over all keyword types.

    Return
    ------
    avg_all_scores, avg_avg_scores, avg_median_scores, avg_max_scores
    """
    all_scores = []
    avg_scores = []
    median_scores = []
    max_scores = []
    min_scores = []

    for keyword in score_dict:
        all_scores.extend(score_dict[keyword])
        avg_scores.append(np.mean(score_dict[keyword]))
        median_scores.append(np.median(score_dict[keyword]))
        max_scores.append(np.max(score_dict[keyword]))
        min_scores.append(np.min(score_dict[keyword]))

    avg_all_scores = np.mean(all_scores)
    avg_avg_scores = np.mean(avg_scores)
    avg_median_scores = np.mean(median_scores)
    avg_max_scores = np.mean(max_scores)
    avg_min_scores = np.mean(min_scores)

    return (
        avg_all_scores, avg_avg_scores, avg_median_scores, avg_max_scores,
        avg_min_scores
        )


def get_spearmanr(prediction_dict, rating_dict, keywords):

    keywords = sorted(keywords)
    utt_keys = sorted(prediction_dict)

    prediction_vector = np.zeros(len(utt_keys)*len(keywords))
    rating_vector = np.zeros(len(utt_keys)*len(keywords))

    i_var = 0
    for utt in utt_keys:
        for keyword in keywords:
            prediction_vector[i_var] = prediction_dict[utt][keyword]
            if keyword in rating_dict[utt]:
                rating_vector[i_var] = rating_dict[utt][keyword]
            else:
                rating_vector[i_var] = 0
            i_var += 1

    return spearmanr(prediction_vector, rating_vector)


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print(f"Reading: {args.json_fn}")
    with open(args.json_fn) as f:
        cost_dict = json.load(f)
    
    captions_fn = Path("data/text")
    print(f"Reading: {captions_fn}")
    captions_dict = get_captions_dict(captions_fn)

    semkeywords_dict_fn = Path("data/semantic_hard.json")
    print(f"Reading: {semkeywords_dict_fn}")
    with open(semkeywords_dict_fn) as f:
        semkeywords_dict = json.load(f)
    semkeywords_counts_dict_fn = Path("data/semantic_counts.json")
    print(f"Reading: {semkeywords_counts_dict_fn}")
    with open(semkeywords_counts_dict_fn) as f:
        semkeywords_counts_dict = json.load(f)

    # Get similarity dict
    keywords = list(set([i.split("_")[0] for i in cost_dict]))
    # print("Keywords:", keywords)
    similarity_dict = {}
    for keyword_key in cost_dict:
        # print(utt_key)
        keyword = keyword_key.split("_")[0]
        for spk_utt_key in cost_dict[keyword_key]:
            utt_key = spk_utt_key[4:]
            if utt_key not in semkeywords_counts_dict:
                continue
            if utt_key not in similarity_dict:
                similarity_dict[utt_key] = {}
            if not keyword in similarity_dict[utt_key]:
                similarity_dict[utt_key][keyword] = []
            similarity_dict[utt_key][keyword].append(
                -cost_dict[keyword_key][spk_utt_key]
                )
    for utt_key in similarity_dict:
        for keyword in similarity_dict[utt_key]:
            similarity_dict[utt_key][keyword] = np.max(
                similarity_dict[utt_key][keyword]
                )

    # Exact QbE
    # print()
    eer_dict, p_at_10_dict, p_at_n_dict = eval_qbe(cost_dict, captions_dict)

    eer_overall, eer_avg, eer_median, eer_max, eer_min = get_avg_scores(
        eer_dict
        )
    p_at_10_overall, p_at_10_avg, p_at_10_median, p_at_10_max, p_at_10_min = (
        get_avg_scores(p_at_10_dict)
        )
    p_at_n_overall, p_at_n_avg, p_at_n_median, p_at_n_max, p_at_n_min = (
        get_avg_scores(p_at_n_dict)
        )
    print()
    print("-"*74)
    print("Exact QbE:")
    print(
        f"EER:  {eer_overall:.4f}, avg: {eer_avg:.4f}, "
        f"median: {eer_median:.4f}, max: {eer_max:.4f}, min: {eer_min:.4f}"
        )
    print(
        f"P@10: {p_at_10_overall:.4f}, avg: {p_at_10_avg:.4f}, "
        f"median: {p_at_10_median:.4f}, max: {p_at_10_max:.4f}, "
        f"min: {p_at_10_min:.4f}"
        )
    print(
        f"P@N:  {p_at_n_overall:.4f}, avg: {p_at_n_avg:.4f}, "
        f"median: {p_at_n_median:.4f}, max: {p_at_n_max:.4f}, "
        f"min: {p_at_n_min:.4f}"
        )
    print("-"*74)

    if len(similarity_dict) == 0:
        return  # no semantic labels for this set

    # Semantic QbE
    print()
    eer_dict, p_at_10_dict, p_at_n_dict = eval_qbe(cost_dict, semkeywords_dict)

    eer_overall, eer_avg, eer_median, eer_max, eer_min = get_avg_scores(
        eer_dict
        )
    p_at_10_overall, p_at_10_avg, p_at_10_median, p_at_10_max, p_at_10_min = (
        get_avg_scores(p_at_10_dict)
        )
    p_at_n_overall, p_at_n_avg, p_at_n_median, p_at_n_max, p_at_n_min = (
        get_avg_scores(p_at_n_dict)
        )
    spearmans_rho = get_spearmanr(
        similarity_dict, semkeywords_counts_dict, keywords
        )

    print()
    print("-"*74)
    print("Semantic QbE:")
    print(
        f"EER:  {eer_overall:.4f}, avg: {eer_avg:.4f}, "
        f"median: {eer_median:.4f}, max: {eer_max:.4f}, min: {eer_min:.4f}"
        )
    print(
        f"P@10: {p_at_10_overall:.4f}, avg: {p_at_10_avg:.4f}, "
        f"median: {p_at_10_median:.4f}, max: {p_at_10_max:.4f}, "
        f"min: {p_at_10_min:.4f}"
        )
    print(
        f"P@N:  {p_at_n_overall:.4f}, avg: {p_at_n_avg:.4f}, "
        f"median: {p_at_n_median:.4f}, max: {p_at_n_max:.4f}, "
        f"min: {p_at_n_min:.4f}"
        )
    print(f"Spearman's rho: {spearmans_rho[0]:.4f}")
    print("-"*74)


if __name__ == "__main__":
    main()
