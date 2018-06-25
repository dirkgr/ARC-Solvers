"""
Script to add support for each answer choice and question, based on the perfect support given in the
OpenBookQA set.
USAGE:
 python scripts/add_gold_retrieved_text.py qa_file output_file

JSONL format of files
 1. qa_file:
 {
    "id":"Mercury_SC_415702",
    "question": {
       "stem":"George wants to warm his hands quickly by rubbing them. Which skin surface will
               produce the most heat?",
       "choices":[
                  {"text":"dry palms","label":"A"},
                  {"text":"wet palms","label":"B"},
                  {"text":"palms covered with oil","label":"C"},
                  {"text":"palms covered with lotion","label":"D"}
                 ]
    },
    "answerKey":"A",
    "fact1": "palms are sweaty",
    "fact2": "knees weak arms are heavy",
  }

 2. output_file:
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
                "text": "palms are sweaty",
                "type": "sentence",
                "ir_pos": 0,
                "ir_score": 100,
            }
    },
     "answerKey":"A"
  }
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text": "dry palms", "label": "A"},
      "support": {
                "text": "knees weak arms are heavy",
                "type": "sentence",
                "ir_pos": 1,
                "ir_score": 100,
            }
    },
     "answerKey":"A"
  }
  ...
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text":"palms covered with lotion","label":"D"}
      "support": {
                "text": "palms are sweaty",
                "type": "sentence",
                "ir_pos": 0,
                "ir_score": 100,
            }
     "answerKey":"A"
  }
  {
   "id": "Mercury_SC_415702",
   "question": {
      "stem": "..."
      "choice": {"text":"palms covered with lotion","label":"D"}
      "support": {
                "text": "knees weak arms are heavy",
                "type": "sentence",
                "ir_pos": 1,
                "ir_score": 100,
            }
     "answerKey":"A"
  }

Every answer choice gets the same support. Every answer choice gets two supports, i.e., the two
facts we get from OpenBookQA.
"""

import json
import os
import sys

from allennlp.common.util import JsonDict
from tqdm._tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))))


def add_retrieved_text(qa_file, output_file):
    with open(output_file, 'w') as output_handle, open(qa_file, 'r') as qa_handle:
        print("Writing to {} from {}".format(output_file, qa_file))
        line_tqdm = tqdm(qa_handle, dynamic_ncols=True)
        for line in line_tqdm:
            json_line = json.loads(line)
            num_hits = 0
            for output_dict in add_facts_to_qajson(json_line):
                output_handle.write(json.dumps(output_dict) + "\n")
                num_hits += 1
            line_tqdm.set_postfix(hits=num_hits)


def add_facts_to_qajson(qa_json: JsonDict):
    fact1 = qa_json["fact1"]
    fact2 = qa_json["fact2"]

    output_dicts_per_question = []
    for choice in qa_json["question"]["choices"]:
        for position, fact in enumerate([fact1, fact2]):
            output_dict_per_fact = create_output_dict(qa_json, choice, fact, position)
            output_dicts_per_question.append(output_dict_per_fact)
    return output_dicts_per_question


# Create the output json dictionary from the QA file json, answer choice json and retrieved HIT
def create_output_dict(qa_json: JsonDict, choice_json: JsonDict, fact: str, position: int):
    output_dict = {
        "id": qa_json["id"],
        "question": {
            "stem": qa_json["question"]["stem"],
            "choice": choice_json,
            "support": {
                "text": fact,
                "type": "sentence",
                "ir_pos": position,
                "ir_score": 100,
            }
        },
        "answerKey": qa_json["answerKey"]
    }
    return output_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Get lines from the ES index and add them to a question set')
    parser.add_argument('question_file', help='Path of file containing questions and answers')
    parser.add_argument('output', help='Name of output file')
    args = parser.parse_args()

    add_retrieved_text(args.question_file, args.output)
