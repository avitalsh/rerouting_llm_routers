import os
import numpy as np
import tiktoken
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import torch
import pickle
import tqdm
import pandas as pd
import openai
import re
import string
import gc


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from RouteLLM.routellm.controller import ModelPair
from RouteLLM.routellm.evals.benchmarks import MTBench
from RouteLLM.routellm.evals.mmlu.domains import ALL_MMLU_DOMAINS

from FastChat.fastchat.llm_judge.common import load_judge_prompts
from FastChat.fastchat.llm_judge.gen_judgment import make_judge_single



def load_emb_model(emb_model, device):
    if emb_model in ["text-embedding-ada-002", "text-embedding-3-small"]:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        encoding = tiktoken.encoding_for_model(emb_model)

        emb_dict = {'emb_model': emb_model, 'encoding_format': "float", 'client': client, 'tokenizer': encoding}
        score_function = cosine_similarity
    else:
        raise ValueError(f"Invalid embedding model {emb_model}")
    return emb_dict, score_function

def get_vocab_and_mask(emb_dict):
    if emb_dict['emb_model'] in ['text-embedding-ada-002', 'text-embedding-3-small']:
        # print(emb_dict['tokenizer'].encode('!'))
        mask_token_id = 0
        mask = emb_dict['tokenizer'].decode([mask_token_id])
        token_vocab = list(np.arange(emb_dict['tokenizer'].n_vocab))
        to_remove = [100256, 100261, 100262, 100263, 100264, 100265, 100266, 100267, 100268, 100269, 100270, 100271, 100272, 100273, 100274, 100275] #this cause panic exception https://github.com/openai/tiktoken/issues/47
        for t in to_remove:
            token_vocab.remove(t)

    else:
        raise ValueError(f"Invalid embedding model {emb_model}")
    return token_vocab, mask_token_id, mask


def compute_token_stats(token_vocab, emb_dict):
    token_vocab = np.array(token_vocab)

    token_counts = [0] * len(token_vocab)

    dataset = load_dataset("wikitext", 'wikitext-103-raw-v1', split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    pbar = tqdm.tqdm(desc=f"Process token stats", total=len(dataloader), leave=False)
    for i, batch in enumerate(dataloader):
        tokens = emb_dict['tokenizer'].encode(batch['text'][0])
        for t in tokens:
            t_idx = np.where(token_vocab == t)[0][0]
            token_counts[t_idx] += 1
        pbar.update(1)

    token_counts = np.array(token_counts)
    token_probs = token_counts / np.sum(token_counts)
    ids = np.argsort(token_counts)[::-1][:50]
    print("50 most popular tokens:")
    for i in ids:
        if token_counts[i] > 0:
            print(
                f"{token_vocab[i]}: {emb_dict['tokenizer'].decode([token_vocab[i]])} ({token_counts[i]} = {token_probs[i]})")

    return token_counts, token_probs


def load_benchmark(benchmark_name):
    if benchmark_name == 'mmlu':
        response_dir = "./RouteLLM/routellm/evals/mmlu/responses"
        all_data = pd.DataFrame()
        for domain in tqdm.tqdm(ALL_MMLU_DOMAINS, desc="Loading domain data"):
            if os.path.exists(f"{response_dir}/mmlu_{domain}_with_answers.csv"):
                cur_data = pd.read_csv(f"{response_dir}/mmlu_{domain}_with_answers.csv")
            else:
                print(f'Preprocess domain {domain} to include answers')
                cur_data = pd.read_csv(f"{response_dir}/mmlu_{domain}.csv")
                cur_data.insert(len(cur_data.columns), "Domain", [domain] * len(cur_data), True)
                cur_data.insert(len(cur_data.columns), "orig_id", np.arange(len(cur_data)), True)
                ds = load_dataset("lighteval/mmlu", domain, split='test')
                cur_data.insert(len(cur_data.columns), "orig_q", ds[:]['question'], True)
                answers_int = ds[:]['answer'] #the answers in numerical form
                answers_letter = np.array(["-"] * len(answers_int)) # the answers in alphabetical form
                num2let = {0: "A", 1: "B", 2: "C", 3: "D"}
                for val in np.unique(answers_int): #convert number to letter
                    mask = (answers_int == val)
                    answers_letter[mask] = num2let[val]
                cur_data.insert(len(cur_data.columns), "answer", answers_letter, True)
                cur_data.insert(len(cur_data.columns), "answers_txt", [ds[i]['choices'][ds[i]['answer']] for i in range(len(ds))], True)
                cur_data.to_csv(f"{response_dir}/mmlu_{domain}_with_answers.csv", index=False)
            all_data = pd.concat(
                [
                    all_data,
                    cur_data,
                ],
                ignore_index=True,
            )
        original_length = len(all_data)

        # Generated using contamination_check.py
        contaminated_prompts = pd.read_json(
            f"./RouteLLM/routellm/evals/mmlu/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"Remaining {len(all_data)}/{original_length} prompts for MMLU after decontamination"
        )
        return all_data

    elif benchmark_name == 'mt-bench':
        return MTBench(ModelPair(strong="", weak=""), []).questions

    elif benchmark_name == 'gsm8k':
        if os.path.exists(f"./RouteLLM/routellm/evals/gsm8k/gsm8k_responses_with_answers.csv"):
            all_data = pd.read_csv(f"./RouteLLM/routellm/evals/gsm8k/gsm8k_responses_with_answers.csv")
        else:
            all_data = pd.read_csv(f"./RouteLLM/routellm/evals/gsm8k/gsm8k_responses.csv")
            ds = load_dataset("openai/gsm8k", "main", split='test')
            all_data.insert(len(all_data.columns), "answer", ds[:]['answer'], True)
            all_data.insert(len(all_data.columns), "orig_q", ds[:]['question'], True)
            all_data.to_csv(f"./RouteLLM/routellm/evals/gsm8k/gsm8k_responses_with_answers.csv", index=False)
        original_len = len(all_data)

        contaminated_prompts = pd.read_json(
            f"./RouteLLM/routellm/evals/gsm8k/contaminated_prompts.jsonl", lines=True
        )["eval_prompt"].tolist()
        all_data = all_data[~all_data["prompt"].isin(contaminated_prompts)]
        print(
            f"{len(all_data)}/{original_len} questions for GSM8K after decontamination."
        )
        return all_data




def get_benchmark_questions_and_ids(benchmark_name, num_samples, seed=0):
    print(f"load dataset {benchmark_name}")
    questions, ids = [], []
    answers, answers_txt = [], []

    if benchmark_name == "mmlu":
        format_prefix = f'Answer the question using the format: "Answer: [A/B/C/D]. Explanation: [EXPLANATION]"\n\n'
        benchmark = load_benchmark(benchmark_name)
        for idx, row in benchmark.iterrows():
            ids.append(idx)
            questions.append(row['prompt'])
            answers.append(row['answer'])
            answers_txt.append(row['answers_txt'])
    elif benchmark_name == "mt-bench":
        format_prefix = ''
        benchmark = load_benchmark(benchmark_name)
        for _, row in benchmark.iterrows():
            ids.append(row['question_id'])
            questions.append(row['turn1'])
    elif benchmark_name == "gsm8k":
        format_prefix = f'Answer the question using the format: "Answer: [Integer number]. Explanation: [EXPLANATION]"\n\n'
        benchmark = load_benchmark(benchmark_name)
        for idx, row in benchmark.iterrows():
            ids.append(idx)
            questions.append(row['prompt'])
            answer_txt = row['answer']
            answer = int(answer_txt.split("####")[-1].strip().replace(",", ""))
            answers.append(answer)
            answers_txt.append(answer_txt)
    else:
        raise ValueError(f"Invalid benchmark {benchmark_name}")

    if num_samples != -1 and len(questions) > num_samples:
        print(f"eval only {num_samples}/{len(questions)} questions")
        questions = np.array(questions)
        answers = np.array(answers)

        if benchmark_name in ['mmlu', 'gsm8k']:
            # we put aside 1000 questions for the threshold calibration, so first remove them before sampling
            np.random.seed(0)
            used_inds = np.random.choice(len(questions), 1000, replace=False)
            questions = np.delete(questions, used_inds)
            answers = np.delete(answers, used_inds)

        np.random.seed(seed)
        inds = np.random.choice(len(questions), num_samples, replace=False)
        questions = np.array(questions)[inds].tolist()
        ids = np.array(ids)[inds].tolist()
        if len(answers) > 0:
            #mt-bench doesn't have GT answers
            answers = np.array(answers)[inds].tolist()
            answers_txt = np.array(answers_txt)[inds].tolist()
        assert len(np.unique(ids)) == len(ids)

    questions = [format_prefix + questions[i] for i in range(len(questions))]

    return questions, ids, answers, answers_txt, format_prefix

def get_model_responses(args, full_modelname, short_modelname, quantize, benchmark_name, details_str, questions):
    model_responses_dir = f"./data/{benchmark_name}/{details_str}/"
    os.makedirs(model_responses_dir, exist_ok=True)
    response_path = os.path.join(model_responses_dir, f"{short_modelname}_q-{quantize}_{len(questions)}_seed{args.seed}.pkl")

    if not os.path.exists(response_path):
        print(f"collect responses for model - {short_modelname} (quantize={quantize}) (saved at:\n{response_path})")
        model_responses = []
        if 'gpt' not in short_modelname:
            device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = AutoTokenizer.from_pretrained(full_modelname)
            if quantize:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(full_modelname,
                                                             quantization_config=quantization_config,
                                                             device_map=device)
            else:
                model = AutoModelForCausalLM.from_pretrained(full_modelname, torch_dtype=torch.float16,
                                                             device_map=device)

            pbar = tqdm.tqdm(desc=f"Get responses", total=len(questions))
            for index, cur_q in enumerate(questions):
                messages = [
                    {"role": "system",
                     "content": "You are a helpful assistant. Respond to the questions as best as you can."},
                    {"role": "user", "content": cur_q},
                ]
                input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                output_ids = model.generate(
                    input_ids,
                    do_sample=False if args.temperature < 1e-4 else True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_response_len,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]):]
                output = tokenizer.decode(output_ids, skip_special_tokens=True)
                if output.startswith('assistant\n\n'):
                    output = output[len('assistant\n\n'):]
                model_responses.append(output)
                pbar.update(1)
                torch.cuda.empty_cache()

            del model
            gc.collect()
            torch.cuda.empty_cache()

        elif 'gpt' in short_modelname:
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            pbar = tqdm.tqdm(desc=f"Get responses", total=len(questions))
            for index, cur_q in enumerate(questions):
                response = client.responses.create(
                    model=short_modelname,
                    instructions="You are a helpful assistant. Respond to the questions as best as you can.",
                    input=cur_q)
                model_responses.append(response.output_text)
                pbar.update(1)

        else:
            raise ValueError(f"Invalid model {full_modelname}")

        with open(response_path, 'wb') as f:
            pickle.dump({'model_responses': model_responses}, f)


        # save a few samples for debugging
        samples_path = response_path.replace('.pkl', '.txt')
        txt = f"10 {benchmark_name} samples for model {short_modelname} (quantize={quantize}). Setting: {details_str}\n\n"
        for i in range(min(10, len(model_responses))):
            txt += f"[{i}]. QUERY:\n{questions[i]}.\n\nANSWER:\n{model_responses[i]}\n\n" + "-"*100 + "\n"
        f = open(samples_path, "w")
        f.write(txt)
        f.close()

    else:
        # print(f"load responses for model - {short_modelname} (quantize={quantize}) (saved at:\n{response_path})")
        with open(response_path, 'rb') as f:
            model_responses = pickle.load(f)['model_responses']

    return model_responses


def confound_query(cur_query_txt, cur_gadget, concat_method='prefix', format_prefix=''):
    if concat_method == "prefix_all":
        confounded_query = cur_gadget + " " + cur_query_txt
    elif concat_method == "prefix": #before query but after formatting prefix
        just_query_txt = cur_query_txt.replace(format_prefix, '')
        confounded_query = format_prefix + cur_gadget + " " + just_query_txt
    elif concat_method == "suffix":
        if bench_name != 'mmlu':
            confounded_query = cur_query_txt + " " + cur_gadget
        else: # mmlu is a multi choice dataset, so we put the suffix by the end of the query and before the list of choices
            just_query_txt = cur_query_txt.replace(format_prefix, '')
            assert "\nA. " in just_query_txt
            confounded_query = format_prefix + just_query_txt.replace("\nA. ", f" {cur_gadget}\nA. ")
    else:
        raise ValueError(f"Invalid concat method {concat_method}")
    return confounded_query



def get_mtbench_scores(questions, responses, judge_model):
    # for mt-bench we use LLM-as-a-judge to compute the benchmark-specific scores
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    judge_prompts = load_judge_prompts('./FastChat/fastchat/llm_judge/data/judge_prompts.jsonl')
    judge = make_judge_single(judge_model, judge_prompts)["default"]
    system_prompt = judge.prompt_template["system_prompt"]

    scores, judgments = [], []
    pbar = tqdm.tqdm(desc=f"Judge responses", total=len(questions))
    for q_idx in range(len(questions)):
        cur_question = questions[q_idx]
        cur_response = responses[q_idx]

        prompt = judge.prompt_template["prompt_template"].format(
            question=cur_question,
            answer=cur_response
        )

        judgment = client.responses.create(
            model=judge_model,
            instructions=system_prompt,
            input=prompt).output_text

        template_1 = "Rating: \[\[([1-9]|10)\]\]"
        template_2 = "**Rating**: \[\[([1-9]|10)\]\]"
        x = re.findall(template_1, judgment)
        if len(x) == 1:
            try:
                cur_score = int(x[0].replace('Rating:', '').replace('[[', '').replace(']]', '').strip())
            except:
                raise Exception(f"error parsing {x[0]}")
        else:
            x = re.findall(template_2, judgment)
            if len(x) == 1:
                try:
                    cur_score = int(x[0].replace('**Rating**:', '').replace('[[', '').replace(']]', '').strip())
                except:
                    raise Exception(f"error parsing {x[0]}")
            else:
                raise Exception(f"no matching template: {judgment}")

        scores.append(cur_score)
        judgments.append(judgment)
        pbar.update(1)

    return np.array(scores), judgments


def get_mmlu_scores(questions, answers, responses):
    # for mmlu we use regex to parse the answer out of the response and compare to the ground truth

    just_letter = "[ABCD][^a-zA-Z0-9]?"
    template_list = [f"Answer:\s{just_letter}", f"The correct answer is\s{just_letter}",
                     f"the correct answer is[\s:\n]+{just_letter}", f"^{just_letter}",
                     f"answer to be:[\s\n]*{just_letter}", f"answer choice {just_letter}",
                     f"is:[\n]*{just_letter}", f"The answer is\s{just_letter}",
                     f"the answer is[\s:\n]+{just_letter}", f"would be {just_letter}",
                     f"Answer:\s\[{just_letter}"]

    pred_answers = []
    pbar = tqdm.tqdm(desc=f"Parse responses", total=len(questions))
    for q_idx in range(len(questions)):
        cur_response = responses[q_idx]
        cur_pred = None
        for t_idx, template in enumerate(template_list):
            x = re.findall(template, cur_response)
            if len(x) == 0:
                # this template doesn't match, move to the next one
                continue
            else:
                if t_idx == 0:
                    cur_pred = x[-1].replace("Answer: ", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 1:
                    cur_pred = x[-1].replace("The correct answer is ", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 2:
                    cur_pred = x[-1].replace("the correct answer is", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 3:
                    cur_pred = x[-1][0]
                elif t_idx == 4:
                    cur_pred = x[-1].replace("answer to be", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 5:
                    cur_pred = x[-1].replace("answer choice", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 6:
                    cur_pred = x[-1].replace("is:", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 7:
                    cur_pred = x[-1].replace("The answer is ", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 8:
                    cur_pred = x[-1].replace("the answer is ", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 9:
                    cur_pred = x[-1].replace("would be ", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")
                elif t_idx == 10:
                    cur_pred = x[-1].replace("Answer: [", "").translate(str.maketrans('', '', string.punctuation)).strip(" ").strip("\n")

                break

        if cur_pred not in ["A", "B", "C", "D"]:
            cur_pred = "F" # if we failed parsing, denote this as wrong answer
            # raise Exception(f"Failed parsing {cur_response}, got {cur_pred}")

        if cur_pred is None:
            raise Exception(f"No template matched {cur_response}")

        pred_answers.append(cur_pred)
        pbar.update(1)

    scores = np.array(pred_answers) == np.array(answers)
    return scores, pred_answers

def parse_num(txt):
    txt = txt.replace(',', '').replace('$', '') #for numbers in the format 1,000 -> 1000
    #fix formatting such that things like 1.2 1,000 1/2 work
    float_num_pattern = "-?[0-9]*[.]?[0-9]+" #1 or 1.2 or .2
    fraction_pattern = f"{float_num_pattern}[\s]?/[\s]?{float_num_pattern}" # 1/1 or 1 / 1 or 1.2 / 1.1
    eq_frac = f"=[\s]?{fraction_pattern}"
    eq_float = f"=[\s]?{float_num_pattern}"
    eq_pattern = f"{eq_frac}|{eq_float}"

    #first look for numbers
    x = re.findall(float_num_pattern, txt)
    if len(x) == 1:
        #there is a single number in the answer, return that number
        return float(x[0])

    elif len(x) > 1:
        if len(x) == 2:
            #check if there is a single fraction
            x = re.findall(fraction_pattern, txt)
            if len(x) == 1:
                #yes there is a single fraction, lets parse it
                frac_sides = x[0].split("/")
                num1 = re.findall(float_num_pattern, frac_sides[0])
                num2 = re.findall(float_num_pattern, frac_sides[1])
                if len(num1) == 1 and len(num2) == 1:
                    if float(num2[0]) != 0:
                        return float(num1[0]) / float(num2[0])
                    else:
                        return None
                else:
                    # if we failed to parse we return none and later mark this as wrong answer
                    return None

        #there are multiple numbers or fractions, check if there is one that matches the format of num\n\n (num can be a float or a fraction)
        x = re.findall(f"{fraction_pattern}\n\n|{float_num_pattern}\n\n", txt)
        if len(x) == 1:
            #there is one number in the right format so take that and ignore other numbers
            return parse_num(x[0]) #this will be recursively parsed as a single number or single fraction
        else:
            #there isn't, lets check for more complicated patterns
            #equations
            x = re.findall(eq_pattern, txt)
            if len(x) >= 1: #there is an equation or multiple so lets take the last one
                pred = parse_num(x[-1]) #this will only take the last "= num" so it will recursively be parsed as a single number or fraction
                return pred
            else: #no equations, but still multiple numbers, so just take the first number or fraction
                x = re.findall(f"{fraction_pattern}|{float_num_pattern}", txt)
                return parse_num(x[0]) # recursively parse first number or fraction
    else:
        # if we failed to parse we return none and later mark this as wrong answer
        return None

def get_gsm8k_scores(questions, answers, responses):
    # for gsm8k we use regex to parse the answer out of the response and compare to the ground truth

    pred_answers = []
    pbar = tqdm.tqdm(desc=f"Parse responses", total=len(questions))
    for q_idx in range(len(questions)):
        cur_response = responses[q_idx].replace(',', '').replace('$', '')  # change things like 1,000 or 1000$ to 1000

        if "Answer:" in cur_response and "Explanation:" in cur_response: # requested format
            options = cur_response.split("Explanation:")
            # we check both before and after "Explanation" if there is an answer
            pred_1 = parse_num(options[0].split("Answer:")[-1])
            pred_2 = parse_num(options[1])
            if pred_1 is not None and pred_2 is not None:  # there are answers in both sides, if one is correct take it
                pred = pred_2 if pred_2 == answers[q_idx] else pred_1
            elif pred_1 is not None:
                pred = pred_1
            else:
                pred = pred_2
        else:
            # not by the format so just try to parse as is
            pred = parse_num(cur_response)

        pred_answers.append(pred)
        pbar.update(1)

    pred_answers = np.array(pred_answers)
    mask = (pred_answers == None)
    wrong_answer = min(answers) - 10
    pred_answers[mask] = wrong_answer # where we didn't manage to parse we just mark as wrong answer

    scores = pred_answers == np.array(answers)
    return scores, pred_answers

def get_ppl_scores(short_modelname, quantize, ppl_model, benchmark_name, details_str, responses, seed=0):
    model_responses_dir = f"./data/{benchmark_name}/{details_str}/"
    scores_path = os.path.join(model_responses_dir, f"ppl_scores_{short_modelname}_q-{quantize}_{ppl_model}_{len(responses)}_seed{seed}.pkl")

    if os.path.exists(scores_path):
        with open(scores_path, 'rb') as f:
            ppl_scores = pickle.load(f)['scores']
    else:
        print("Compute perplexity scores")
        if ppl_model == 'gpt2':
            model = AutoModelForCausalLM.from_pretrained("gpt2", device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        else:
            raise ValueError(f"invalid ppl model {ppl_model}")

        device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        ppl_scores = []
        pbar = tqdm.tqdm(desc=f"Compute PPL", total=len(responses))
        for q_idx in range(len(responses)):
            cur_response = responses[q_idx]
            inputs = tokenizer(cur_response, return_tensors="pt", truncation=True).to(device)
            loss = model(input_ids=inputs["input_ids"], labels=inputs["input_ids"]).loss
            ppl = torch.exp(loss).item()
            ppl_scores.append(ppl)
            pbar.update(1)

        del model
        gc.collect()
        torch.cuda.empty_cache()

        ppl_scores = np.array(ppl_scores)
        with open(scores_path, 'wb') as f:
            pickle.dump({'scores': ppl_scores}, f)

    return ppl_scores

def get_benchmark_specific_scores(short_modelname, quantize, benchmark_name, details_str, questions, answers, responses, judge_model='', seed=0):
    model_responses_dir = f"./data/{benchmark_name}/{details_str}/"
    scores_path = os.path.join(model_responses_dir, f"bench_scores_{short_modelname}_q-{quantize}_{len(questions)}_seed{seed}.pkl")

    if os.path.exists(scores_path):
        with open(scores_path, 'rb') as f:
            scores = pickle.load(f)['scores']

    else:
        print("Compute benchmark-specific scores")
        if benchmark_name == 'mt-bench':
            scores, judgments = get_mtbench_scores(questions, responses, judge_model)
            with open(scores_path, 'wb') as f:
                pickle.dump({'scores': scores, 'judgments': judgments}, f)
        elif benchmark_name == 'mmlu':
            scores, pred_answers = get_mmlu_scores(questions, answers, responses)
            with open(scores_path, 'wb') as f:
                pickle.dump({'scores': scores, 'pred_answers': pred_answers}, f)
        elif benchmark_name == 'gsm8k':
            scores, pred_answers = get_gsm8k_scores(questions, answers, responses)
            with open(scores_path, 'wb') as f:
                pickle.dump({'scores': scores, 'pred_answers': pred_answers}, f)
        else:
            raise ValueError(f"Invalid benchmark {benchmark_name}")

    return scores


def get_calibrated_threshold(benchmark_name, router_name, router, strong_model_pct=0.5, use_arena=False, num_calib=1000):

    if use_arena or benchmark_name == "mt-bench":
        thresholds_df = load_dataset(
            "routellm/lmsys-arena-human-preference-55k-thresholds", split="train"
        ).to_pandas()
        threshold = thresholds_df[router_name].quantile(q=1 - strong_model_pct)
    else:
        threshold_path = f'./data/{benchmark_name}/thresholds/'
        os.makedirs(threshold_path, exist_ok=True)
        threshold_path = os.path.join(threshold_path, f'{router_name}_{strong_model_pct}_{num_calib}.pkl')

        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                threshold = pickle.load(f)['threshold']
        else:
            print(f"calibrate threshold on {num_calib} scores")
            #sample calibration set
            all_questions, _, _, _, _ = get_benchmark_questions_and_ids(benchmark_name=benchmark_name, num_samples=-1)
            np.random.seed(0)
            inds = np.random.choice(len(all_questions), num_calib, replace=False)

            #get scores for calibration set
            calib_scores = [router.calculate_strong_win_rate(all_questions[i]) for i in inds]

            threshold = np.quantile(calib_scores, 1 - strong_model_pct)
            with open(threshold_path, 'wb') as f:
                pickle.dump({'threshold': threshold, 'calib_scores': calib_scores}, f)

    return threshold




