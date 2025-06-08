import argparse
import json
import os
import torch
import transformers

from datasets import load_dataset
from huggingface_hub import login
from tqdm import tqdm


values_prompt_context = {
    "Self-direction": {
        "description": "Independent thought and action—choosing, creating, and exploring",
        "Self-direction: thought": "It is good to have own ideas and interests.\nContained values and associated arguments (examples):\n\nBe creative: arguments towards more creativity or imagination\nBe curious: arguments towards more curiosity, discoveries, or general interestingness\nHave freedom of thought: arguments toward people figuring things out on their own, towards less censorship, or towards less influence on thoughts",
        "Self-direction: action": "It is good to determine one's own actions.\nContained values and associated arguments (examples):\n\nBe choosing own goals: arguments towards allowing people to choose what is best for them, to decide on their life, and to follow their dreams\nBe independent: arguments towards allowing people to plan on their own and not ask for consent\nHave freedom of action: arguments towards allowing people to be self-determined and able to do what they want\nHave privacy: arguments towards allowing for private spaces, time alone, and less surveillance, or towards more control on what to disclose and to whom",
    },
    "Stimulation": "It is good to experience excitement, novelty, and change.\nHave an exciting life: arguments towards allowing people to experience foreign places and special activities or having perspective-changing experiences\nHave a varied life: arguments towards allowing people to engage in many activities and change parts of their life or towards promoting local clubs (sports, ...)\nContained values and associated arguments (examples):\n\nBe daring: arguments towards more risk-taking",
    "Hedonism": "It is good to experience pleasure and sensual gratification.\nContained values and associated arguments (examples):\n\nHave pleasure: arguments towards making life enjoyable or providing leisure, opportunities to have fun, and sensual gratification",
    "Achievement": "It is good to be successful in accordance with social norms.\nContained values and associated arguments (examples):\n\nBe ambitious: arguments towards allowing for ambitions and climbing up the social ladder\nHave success: arguments towards allowing for success and recognizing achievements\nBe capable: arguments towards acquiring competence in certain tasks, being more effective, and showing competence in solving tasks\nBe intellectual: arguments towards acquiring high cognitive skills, being more reflective, and showing intelligence\nBe courageous: arguments towards being more courageous and having people stand up for their beliefs",
    "Power": {
        "description": "Social status and prestige, control or dominance over people and resources",
        "Power: dominance": "It is good to be in positions of control over others.\nContained values and associated arguments (examples):\n\nHave influence: arguments towards having more people to ask for a favor, more influence, and more ways to control events\nHave the right to command: arguments towards allowing the right people to take command, putting experts in charge, and clearer hierarchies of command, or towards fostering leadership",
        "Power: resources": "It is good to have material possessions and social resources.\nContained values and associated arguments (examples):\n\nHave wealth: arguments towards allowing people to gain wealth and material possession, show their wealth, and exercise control through wealth, or towards financial prosperity",
    },

    "Face": "It is good to maintain one's public image.\nContained values and associated arguments (examples):\n\nHave social recognition: arguments towards allowing people to gain respect and social recognition or avoid humiliation\nHave a good reputation: arguments towards allowing people to build up their reputation, protect their public image, and spread reputation",
    "Security": {
        "description": "Safety, harmony, and stability of society, of relationships, and of self",
        "Security: personal": "It is good to have a secure immediate environment.\nContained values and associated arguments (examples):\n\nHave a sense of belonging: arguments towards allowing people to establish, join, and stay in groups, show their group membership, and show that they care for each other, or towards fostering a sense of belonging\nHave good health: arguments towards avoiding diseases, preserving health, or having physiological and mental well-being\nHave no debts: arguments towards avoiding indebtedness and having people return favors\nBe neat and tidy: arguments towards being more clean, neat, or orderly\nHave a comfortable life: arguments towards providing subsistence income, having no financial worries, and having a prosperous life, or towards resulting in a higher general happiness",
        "Security: societal": "It is good to have a secure and stable wider society.\nContained values and associated arguments (examples):\n\nHave a safe country: arguments towards a state that can better act on crimes, and defend or care for its citizens, or towards a stronger state in general\nHave a stable society: arguments towards accepting or maintaining the existing social structure or towards preventing chaos and disorder at a societal level",
    },
    "Tradition": "It is good to maintain cultural, family, or religious traditions.\nContained values and associated arguments (examples):\n\nBe respecting traditions: arguments towards allowing to follow one's family's customs, honoring traditional practices, maintaining traditional values and ways of thinking, or promoting the preservation of customs\nBe holding religious faith: arguments towards allowing the customs of a religion and to devote one's life to their faith, or towards promoting piety and the spreading of one's religion",
    "Conformity": {
        "description": "Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations or norms",
        "Conformity: rules": "It is good to comply with rules, laws, and formal obligations.\nContained values and associated arguments (examples):\n\nBe compliant: arguments towards abiding to laws or rules and promoting to meet one's obligations or recognizing people who do\nBe self-disciplined: arguments towards fostering to exercise restraint, follow rules even when no-one is watching, and to set rules for oneself\nBe behaving properly: arguments towards avoiding to violate informal rules or social conventions or towards fostering good manners",
        "Conformity: interpersonal": "It is good to avoid upsetting or harming others.\nContained values and associated arguments (examples):\n\nBe polite: arguments towards avoiding to upset other people, taking others into account, and being less annoying for others\nBe honoring elders: arguments towards following one's parents or showing faith and respect towards one's elders\n",
    },
    "Humility": "It is good to recognize one's own insignificance in the larger scheme of things.\nContained values and associated arguments (examples):\n\nBe humble: arguments towards demoting arrogance, bragging, and thinking one deserves more than other people, or towards emphasizing the successful group over single persons and giving back to society\nHave life accepted as is: arguments towards accepting one's fate, submitting to life's circumstances, and being satisfied with what one has",
    "Benevolence": {
        "description": "Preserving and enhancing the welfare of those with whom one is in frequent personal contact (the 'in-group')",
        "Benevolence: caring": "It is good to work for the welfare of one's group's members.\nContained values and associated arguments (examples):\n\nBe helpful: arguments towards helping the people in one's group and promoting to work for the welfare of others in one group\nBe honest: arguments towards being more honest and recognizing people for their honesty\nBe forgiving: arguments towards allowing people to forgive each other, giving people a second chance, and being merciful, or towards providing paths to redemption\nHave the own family secured: arguments towards allowing people to have, protect, and care for their family\nBe loving: arguments towards fostering close relationships and placing the well-being of others above the own, or towards allowing to show affection, compassion, and sympathy",
        "Benevolence: dependability": "It is good to be a reliable and trustworthy member of one's group.\nContained values and associated arguments (examples):\n\nBe responsible: arguments towards clear responsibilities, fostering confidence, and promoting reliability\nHave loyalty towards friends: arguments towards being a dependable, trustworthy, and loyal friend, or towards allowing to give friends a full backing",
    },
    "Universalism": {
        "description": "Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature",
        "Universalism: concern": "It is good to strive for equality, justice, and protection for all people.\nContained values and associated arguments (examples):\n\nHave equality: arguments towards fostering people of a lower social status, helping poorer regions of the world, providing all people with equal opportunities in life, and resulting in a world were success is less determined by birth\nBe just: arguments towards allowing justice to be 'blind' to irrelevant aspects of a case, promoting fairness in competitions, protecting the weak and vulnerable in society, and resulting a world were people are less discriminated based on race, gender, and so on, or towards fostering a general sense for justice\nHave a world at peace: arguments towards nations ceasing fire, avoiding conflicts, and ending wars, or promoting to see peace as fragile and precious or to care for all of humanity",
        "Universalism: nature": "It is good to preserve the natural environment.\nContained values and associated arguments (examples):\n\nBe protecting the environment: arguments towards avoiding pollution, fostering to care for nature, or promoting programs to restore nature\nHave harmony with nature: arguments towards avoiding chemicals and genetically modified organisms (especially in nutrition), or towards treating animals and plants like having souls, promoting a life in harmony with nature, and resulting in more people reflecting the consequences of their actions towards the environment\nHave a world of beauty: arguments towards allowing people to experience art and stand in awe of nature, or towards promoting the beauty of nature and the fine arts",
        "Universalism: tolerance": "It is good to accept and try to understand those who are different from oneself.\nContained values and associated arguments (examples):\n\nBe broadminded: arguments towards allowing for discussion between groups, clearing up with prejudices, listening to people who are different from oneself, and promoting to life within a different group for some time, or towards promoting tolerance between all kinds of people and groups in general\nHave the wisdom to accept others: arguments towards allowing people to accept disagreements and people even when one disagrees with them, to promote a mature understanding of different opinions, or to decrease partisanship or fanaticism",
        "Universalism: objectivity": "It is good to search for the truth and think in a rational and unbiased way.\nContained values and associated arguments (examples):\n\nBe logical: arguments towards going for the numbers instead of gut feeling, towards a rational, focused, and consistent way of thinking, towards a rational analysis of circumstances, or towards promoting the scientific method\nHave an objective view: arguments towards fostering to seek the truth, to take on a neutral perspective, to form an unbiased opinion, and to weigh all pros and cons, or towards providing people with the means to make informed decisions"
    }
}

# Fixed order of annotation keys
annotation_keys = [
    "Self-direction: thought",
    "Self-direction: action",
    "Stimulation",
    "Hedonism",
    "Achievement",
    "Power: dominance",
    "Power: resources",
    "Face",
    "Security: personal",
    "Security: societal",
    "Tradition",
    "Conformity: rules",
    "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring",
    "Benevolence: dependability",
    "Universalism: concern",
    "Universalism: nature",
    "Universalism: tolerance",
    "Universalism: objectivity"
]


def prompt_model(pipeline, terminators, argument, value, values_description, fewshot_examples, model):
    """
    Prompt the language model to evaluate if a given human value is influencing an argument.

    Parameters:
        pipeline (Pipeline): The prediction pipeline to use for evaluation.
        terminators (list): List of token IDs that indicate the end of the response.
        argument (dict): The argument to evaluate, containing the premise.
        value (str): The value to evaluate.
        values_description (str): Description of the value category.
        fewshot_examples (dict): Dictionary containing few-shot examples for each value.
        model (str): The model to use.

    Returns:
        str: The model's response indicating whether the value is influencing the argument ("yes" or "no").
    """
    system_prompt = (
        "I will give you the premise of an argument along with a description of a human value. It is your task to assess if the human value is influencing the argument."
    )

    user_prompt = (
        f"Premise: {argument['Premise']}\n"
        #f"Human value category: {value}\n"
        f"Description of the value category: {values_description}\n\n"
        f"Given the premise and the description of the human value category, classify whether the argument relies on that category or not.\n\n"
        f"IMPORTANT:\n"
        f"Only respond with one word: 'yes' or 'no'.\n"
        f"No explanations.\n"
        f"If the human value is explicitly or implicitly present in the argument, answer 'yes'.\n"
        f"If not present at all, answer 'no'.\n\n"
    )
    if fewshot_examples:
        user_prompt += "Here are some examples:\n\n"
        for example in fewshot_examples[value]:
            user_prompt += (
                f"Premise: {example[0]}\n"
                f"Answer: {example[1]}\n\n"
            )

    #user_prompt += (
    #    f"Premise: {argument['Premise']}\n"
    #    "Answer: "
    #)

    prompt = (
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_prompt}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    if model == "llama":
        response = outputs[0]["generated_text"].split("<|end_header_id|>")[-1].strip().lower()
    else:
        response = outputs[0]["generated_text"].split('</think>')[-1].strip().lower()
        if "yes" in response:
            response = "yes"
    print(f"{value}: {response}")
    breakpoint()
    return response


def get_examples(train, value):
    """
    Get a few-shot example set for the given value from the training data.

    Parameters:
        train (Dataset): The training dataset containing arguments and their labels.
        value (str): The value for which to retrieve examples.

    Returns:
        list: A list of tuples containing the premise and the label ("yes" or "no") for the specified value.

    """
    examples = []
    value_index = annotation_keys.index(value)
    yes_count = 0
    no_count = 0
    for argument in train:
        label = argument["Labels"][value_index]
        if label == 1 and yes_count < 3:
            examples.append((argument["Premise"], "yes"))
            yes_count += 1
        elif label == 0 and no_count < 3:
            examples.append((argument["Premise"], "no"))
            no_count += 1
        if yes_count >= 3 and no_count >= 3:
            break

    return examples

def argparser():
    """
    Argument parser for the value evaluation task.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="validation",
        choices=["train", "validation", "test"],
        help="Which data split to use"
    )
    parser.add_argument(
        "--model", type=str, default="llama",
        choices=["llama", "deepseek"],
        help="Which model to run"
        )
    parser.add_argument(
        "--fewshot", action="store_true",
        help="Whether to use few-shot examples in the prompt"
    )
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="Hugging Face token for authentication (optional, can also use environment variable HF_TOKEN)"
    )
    return parser.parse_args()

def main():
    """
    Main function to run the value evaluation task using a language model.
    """
    args = argparser()
    login(args.hf_token)  # Or use os.environ["HF_TOKEN"]
    models = {
        "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
        "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        }

    pipeline = transformers.pipeline(
        "text-generation",
        model=models[args.model],
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = load_dataset("webis/Touche23-ValueEval", trust_remote_code=True)
    train = dataset['train']
    split_data = dataset[args.split]

    train = train.shuffle(seed=10)

    fewshot_examples = {}
    if args.fewshot:
        for value in annotation_keys:
            fewshot_examples[value] = get_examples(train, value)
        for value in values_prompt_context.keys():
            vs = [v for v in annotation_keys if v.startswith(value)]
            fewshot_examples[value] = [ex for v in vs for ex in fewshot_examples[v]]

    results = {}

    for argument in tqdm(split_data):
        annotations = {key: 0 for key in annotation_keys}

        for value in values_prompt_context.keys():
            try:
                value_description = values_prompt_context[value]["description"]
                response = prompt_model(pipeline, terminators, argument, value, value_description, fewshot_examples,
                                        args.model)
                if response == "yes":
                    for sub_value in values_prompt_context[value].keys():
                        if sub_value == "description":
                            continue
                        sub_description = values_prompt_context[value][sub_value]
                        sub_response = prompt_model(pipeline, terminators, argument, sub_value, sub_description,
                                                    fewshot_examples, args.model)
                        if sub_response == "yes":
                            annotations[sub_value] = 1
            except TypeError:
                value_description = values_prompt_context[value]
                response = prompt_model(pipeline, terminators, argument, value, value_description, fewshot_examples,
                                        args.model)
                if response == "yes":
                    annotations[value] = 1
        arg_id = argument["Argument ID"]
        results[arg_id] = {
            "correct_labels": argument["Labels"],
            "predicted": [annotations[key] for key in annotation_keys]
            }
        print(f"{argument['Labels']}\n {[annotations[key] for key in annotation_keys]}")

    # Save to JSON
    shot = "oneshot"
    if args.fewshot:
        shot = "fewshot"

    filename = f"predictions_{args.split}_{args.model}_{shot}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()