import argparse
import datetime
import json
import os
import uuid

import faiss

from PromptManager import Prompt

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--create", action="store_true")
parser.add_argument("--eval", action="store_true")

CURRENT_INDEX_NAME = "main"

# Initialize prompt data
prompt_data = Prompt()

# Extract required information from prompt data
index_number = prompt_data.index_id
queried_index = prompt_data.index_name
prompt_id = prompt_data.prompt_id

# Evaluation prompt template
EVALUATE_PROMPT = """
You are a bot tasked with verifying whether an answer is substantiated by a Source listed below.

If a Source proves the answer is correct, respond with the word "CORRECT";
if it proves the answer is incorrect, respond with the word "INCORRECT";
if you are unsure, respond with "UNSURE".

Claim:
[[[QUERY]]]

Sources:
[[[SOURCES]]]
"""

# Load vector index
vector_index = faiss.read_index(f"indices/{index_number}/{queried_index}_vector_index.bin")

# Load schema
with open(f"indices/{index_number}/{queried_index}_schema.json", "r") as f:
    schema = json.load(f)


class Evaluation:
    def __init__(self):
        # Load existing evaluations from files
        self.evals = []
        evals_directory = "evals"
        if os.path.exists(evals_directory):
            for file in os.listdir(evals_directory):
                if file.endswith(".json"):
                    with open(os.path.join(evals_directory, file), "r") as f:
                        self.evals.extend(json.load(f))

        self.successful_evals = []
        self.failed_evals = []
        self.unsure_evals = []
        self.eval_started_time = None
        self.eval_ended_time = None
        self.uuid = str(uuid.uuid4())
        self.stats = {}

    def run_evals(self):
        # Start evaluation timer
        self.eval_started_time = datetime.datetime.now()

        for count, eval in enumerate(self.evals):
            print(f"Running eval \"{eval['question']}\" ({count + 1}/{len(self.evals)})")
            facts = []

            # Get facts, sources, and knn for the evaluation question
            facts_and_sources_text, knn, references = prompt_data.get_facts_and_knn(
                eval["question"], vector_index, schema, facts
            )

            current_date = datetime.datetime.now().strftime("%Y-%m-%d")

            # Execute prompt with provided inputs
            response = prompt_data.execute(
                {
                    "CURRENT_DATE": current_date,
                    "FACTS": "\n".join(facts),
                    "SOURCES": facts_and_sources_text,
                    "QUERY": eval["question"],
                }
            )

            print(response)

            print("Evaluating response...\n\n\n")

            # Evaluate the response using the EVALUATE_PROMPT
            eval_response = prompt_data.execute(
                {
                    "QUERY": response,
                    "SOURCES": facts_and_sources_text,
                },
                prompt_text=EVALUATE_PROMPT,
                temperature=0.0,
            )

            print(eval_response)

            eval_record = {
                "response": eval_response,
                "knn": knn,
                "references": references,
                "question": eval["question"],
            }

            # Categorize the evaluation response
            if "CORRECT" in eval_response:
                self.successful_evals.append(eval_record)
            elif "INCORRECT" in eval_response:
                self.failed_evals.append(eval_record)
            elif "UNSURE" in eval_response:
                self.unsure_evals.append(eval_record)

        # Stop evaluation timer
        self.eval_ended_time = datetime.datetime.now()

    def get_eval_stats(self):
        # Calculate precision, recall, and F1 score
        precision, recall, f1_score = self.calculate_f1_score()

        self.stats = {
            "successful_evals_count": len(self.successful_evals),
            "failed_evals_count": len(self.failed_evals),
            "unsure_evals_count": len(self.unsure_evals),
            "successful_evals": self.successful_evals,
            "failed_evals": self.failed_evals,
            "unsure_evals": self.unsure_evals,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "eval_started_time": self.eval_started_time.strftime("%Y-%m-%d %H:%M:%S"),
            "eval_ended_time": self.eval_ended_time.strftime("%Y-%m-%d %H:%M:%S"),
            "uuid": self.uuid,
        }

        return self.stats

    def pretty_print_eval_stats(self):
        # Print evaluation statistics
        stats = self.stats
        print(f"Precision: {stats['precision']}")
        print(f"Recall: {stats['recall']}")
        print(f"F1 Score: {stats['f1_score']}")
        print(f"Successful evals: {stats['successful_evals_count']}")
        print(f"Failed evals: {stats['failed_evals_count']}")
        print(f"Unsure evals: {stats['unsure_evals_count']}")
        print(f"Eval started at: {stats['eval_started_time']}")

    def calculate_f1_score(self):
        if len(self.successful_evals) == 0:
            return 0, 0, 0

        precision = len(self.successful_evals) / (
            len(self.successful_evals) + len(self.failed_evals)
        )
        recall = len(self.successful_evals) / (
            len(self.successful_evals) + len(self.unsure_evals)
        )
        f1_score = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1_score

    def save_evals(self):
        evals_file = "evals.json"

        if not os.path.exists(evals_file):
            with open(evals_file, "w") as f:
                json.dump([], f)

        with open(evals_file, "r") as f:
            all_evals = json.load(f)

        eval_report = {
            "prompt_id": prompt_id,
            "index_id": index_number,
            "eval_uuid": self.uuid,
            "index_name": queried_index,
            "generated_on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "stats": self.get_eval_stats(),
        }

        all_evals.append(eval_report)

        with open(evals_file, "w") as f:
            json.dump(all_evals, f)

    def create_eval(self, question, answer, eval_name):
        evals_directory = "evals"
        if not os.path.exists(evals_directory):
            os.mkdir(evals_directory)

        eval = {"question": question, "answer": answer}

        to_write = [eval]

        eval_file_path = os.path.join(evals_directory, f"{eval_name}.json")

        if os.path.exists(eval_file_path):
            with open(eval_file_path, "r") as f:
                evals = json.load(f)

            evals.append(eval)
            to_write = evals

        with open(eval_file_path, "w") as f:
            json.dump(to_write, f)

    def create_eval_interactive(self, eval_name=None):
        question = input("Question: ")
        answer = input("Answer: ")

        if eval_name is None:
            eval_name = input("Eval name: ")

        self.create_eval(question, answer, eval_name)


if __name__ == "__main__":
    eval = Evaluation()

    if parser.parse_args().create:
        while True:
            eval.create_eval_interactive("coffee")

    if parser.parse_args().eval:
        eval.run_evals()
        eval.pretty_print_eval_stats()
        eval.save_evals()
