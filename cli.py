import argparse

import text_analyser


parser = argparse.ArgumentParser()

parser.add_argument("-f", "--french",
                    help="French text to compare",
                    type=str,
                    required=True)

parser.add_argument("-e", "--english",
                    help="English text to compare",
                    type=str,
                    required=True)

parser.add_argument("-a", "--algorithm",
                    help="Similarity algorithm (default: cosinus)",
                    type=str,
                    choices=set(("cosinus", "jaccard")),
                    default="cosinus")

args = parser.parse_args()

analyser = text_analyser.TextAnalyser()

algos = {
    'cosinus': analyser.cos_similarity,
    'jaccard':  analyser.jaccard_similarity
    'levenshtein': analyser.levenshtein_similarity
}

is_it = analyser.compare(
    args.english, args.french, algos[args.algorithm])

print(is_it)
