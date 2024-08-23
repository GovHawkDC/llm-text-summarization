# Python built-in module
import json
import pathlib

import click

# Python installed module
import tiktoken
import yaml

from .cluster_summarization import ClusterBasedSummary

# Python user defined module
from .cod import COD
from .map_reduce import MapReduce


def validate_configs(config_dict):
    validation_config_file_name = "validation_config.yaml"
    # Load the validation config file
    with open(validation_config_file_name) as yaml_obj:
        validation_config_dict = yaml.safe_load(yaml_obj)

    max_num_closest_points_per_cluster = validation_config_dict[
        "max_num_closest_points_per_cluster"
    ]
    max_medium_token_length = validation_config_dict["max_medium_token_length"]
    llm_token_mapping = validation_config_dict["llm_token_mapping"]

    success_status_message = "SUCCESS"
    failure_status_message = "FAILED"

    if (
        config_dict["sentence_splitter"]["approx_total_doc_tokens"]
        + config_dict["sentence_splitter"]["tolerance_limit_tokens"]
        >= llm_token_mapping[config_dict["embedding"]["model_name"]]
    ):
        print(
            "[ERROR] The sum of `approx_total_doc_tokens` and `tolerance_limit_tokens` in `sentence_splitter` config should not exceed the value of {} model's token limit!".format(
                config_dict["embedding"]["model_name"]
            )
        )
        return failure_status_message
    if (
        config_dict["summary_type_token_limit"]["short"]
        + config_dict["cod"]["max_tokens"]
        >= llm_token_mapping[config_dict["cod"]["model_name"]]
    ):
        print(
            "[ERROR] The sum of `short` and `max_tokens` in `summary_type_token_limit` and `cod` config respectively should not exceed the value of {} model's token limit!".format(
                config_dict["cod"]["model_name"]
            )
        )
        return failure_status_message
    if (
        config_dict["cluster_summarization"]["num_closest_points_per_cluster"]
        > max_num_closest_points_per_cluster
    ):
        print(
            "[ERROR] The `num_closest_points_per_cluster` in `cluster_summarization` config should not be more than {}".format(
                max_num_closest_points_per_cluster
            )
        )
        return failure_status_message
    if config_dict["summary_type_token_limit"]["medium"] > max_medium_token_length:
        print(
            "[ERROR] The `medium` in `summary_type_token_limit` config should not be more than {} tokens.".format(
                max_medium_token_length
            )
        )
        return failure_status_message
    return success_status_message


@click.command()
@click.argument(
    "input_file", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument("output_file", type=click.Path(writable=True, dir_okay=False))
def main(input_file, output_file):

    input_file_path = pathlib.Path(input_file)
    output_file_path = pathlib.Path(output_file)

    config_file_name = "config.yaml"

    # Load the config file
    with open(config_file_name) as yaml_obj:
        config_dict = yaml.safe_load(yaml_obj)

    # Validate the configuarations
    validation_status = validate_configs(config_dict)
    if validation_status == "FAILED":
        return

    with input_file_path.open() as input_obj:
        text_content = input_obj.read()

    # Identify the summary type, short, medium or long form text summarization
    tiktoken_encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    total_tokens = len(tiktoken_encoding.encode(text_content))
    total_words = int(total_tokens * 0.75)

    print(
        "[INFO] Total tokens in the input text to be summarized: {}".format(
            total_tokens
        )
    )
    print(
        "[INFO] Approx total words in the input text to be summarized: {}".format(
            total_words
        )
    )

    if total_tokens <= config_dict["summary_type_token_limit"]["short"]:
        print("[INFO] The type of summary is for short-formed text.")
        chain_of_density_summarizer = COD(config_dict)
        result_dict = chain_of_density_summarizer(text_content)
    elif total_tokens <= config_dict["summary_type_token_limit"]["medium"]:
        print("[INFO] The type of summary is for medium sized text.")
        map_reduce_summarizer = MapReduce(config_dict)
        result_dict = map_reduce_summarizer(text_content)
    else:
        print("[INFO] The type of summary is for long-formed text.")
        cluster_summarizer = ClusterBasedSummary(config_dict)
        result_dict = cluster_summarizer(text_content)

    if isinstance(result_dict, dict):
        with output_file_path.open("w") as json_obj:
            json.dump(result_dict, json_obj, indent=4)
        print("[INFO] Summary output is successfully created in the output folder!")


if __name__ == "__main__":
    main()
