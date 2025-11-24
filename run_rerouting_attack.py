import copy
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RouteLLM"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FastChat"))

import random
import numpy as np
import torch
import tqdm
import pickle

from RouteLLM.routellm.routers.routers import ROUTER_CLS
from RouteLLM.routellm.controller import GPT_4_AUGMENTED_CONFIG, ModelPair

import utils

from huggingface_hub import login
login(token=os.environ["HF_KEY"])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate routers on various benchmarks.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--router",
        # nargs="+",
        type=str,
        default="bert",
        choices=list(ROUTER_CLS.keys()),
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=[
            "mmlu",
            "mt-bench",
            "gsm8k",
        ],
        default="mt-bench",
    )
    parser.add_argument("--strong-model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--weak-model", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1")
    parser.add_argument('--quantize_strong', action='store_true')
    parser.add_argument('--quantize_weak', action='store_true')
    parser.add_argument("--emb_model", default='text-embedding-3-small', choices=['text-embedding-3-small'])
    parser.add_argument("--ppl_model", default='gpt2', choices=['gpt2'])
    parser.add_argument("--judge_model", default='gpt-4o', choices=['gpt-4o'])
    parser.add_argument("--concat_method", default='prefix', choices=['suffix', 'prefix', 'prefix_all'])
    parser.add_argument("--early_stop", type=int, default=25)
    parser.add_argument("--token_sampling", default='uniform', choices=['uniform', 'probs', 'probs_filter_top'])
    parser.add_argument("--gadget_init", default='mask', choices=['random', 'mask'])
    parser.add_argument("--num_tokens", type=int, default=10)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_attacks", type=int, default=10)
    parser.add_argument("--strong_model_pct", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument('--max_response_len', type=int, default=1024)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    ####################################################################################################################
    ############### Model loading ###############

    config = GPT_4_AUGMENTED_CONFIG
    print("-"*100 + f"\nLoad router: {args.router}")
    routed_pair = ModelPair(strong=args.strong_model, weak=args.weak_model)
    router = ROUTER_CLS[args.router](**config.get(args.router, {}))

    print("-"*100 + f"\nLoad embedding model: {args.emb_model}")
    emb_dict, score_func = utils.load_emb_model(emb_model=args.emb_model, device=device)
    token_vocab, mask_token_id, mask = utils.get_vocab_and_mask(emb_dict)
    print(f"token_vocab size = {len(token_vocab)}. mask_token_id={mask_token_id}, mask={mask}")
    if 'probs' in args.token_sampling:
        token_probs_path = f"./data/tokenizers/{args.emb_model}_wikitext_stats.pkl"
        if not os.path.exists(token_probs_path):
            os.makedirs("./data/tokenizers/", exist_ok=True)
            print(f"    preprocess token stats using wikitext")
            token_counts, token_probs = utils.compute_token_stats(token_vocab, emb_dict)
            with open(token_probs_path, 'wb') as f:
                pickle.dump({'token_probs': token_probs, 'token_counts': token_counts}, f)

        with open(token_probs_path, 'rb') as f:
            token_stats = pickle.load(f)
        token_probs = token_stats['token_probs']
        token_counts = token_stats['token_counts']

        if args.token_sampling == 'probs_filter_top':
            ids = np.argsort(token_counts)[::-1][:100]
            token_vocab = np.delete(token_vocab, ids)
            token_counts = np.delete(token_counts, ids)
            token_probs = token_counts / np.sum(token_counts)

    ####################################################################################################################
    ############### Gadget generation ###############

    print("-"*100 + f"\nCreating {args.num_attacks} query-free confounder gadgets")

    opt_details = f'emb_{args.emb_model}_init_{args.gadget_init}_ts_{args.token_sampling}_nt_{args.num_tokens}_i_{args.num_iterations}_es_{args.early_stop}_bs_{args.batch_size}'
    gadget_dir = f'./confounder_gadgets/{args.router}/{opt_details}'
    os.makedirs(gadget_dir, exist_ok=True)
    for gadget_id in range(args.num_attacks):
        gadget_path = os.path.join(gadget_dir, f'gadget_{gadget_id}.pkl')
        if os.path.exists(gadget_path):
            print(f"loading existing gadget {gadget_id}/{args.num_attacks}")
            with open(gadget_path, 'rb') as f:
                gadget_data = pickle.load(f)
        else:
            print(f"generate from scratch gadget {gadget_id}/{args.num_attacks}")
            gadget_data = {'gadget_hist': [], 'score_hist': [], 'loss_hist': [],
                           'sampled_token_hist': [], 'rep_loc_hist': [], 'idx_chosen_sampled_token_hist': [], 'val_chosen_sampled_token_hist': [], #for debugging
                           'stop_iter': -1}
            # to make sure we don't get the same gadgets on different runs, we use the index as the seed as well
            random.seed(gadget_id + args.seed)
            np.random.seed(gadget_id + args.seed)
            rng = np.random.default_rng(gadget_id + args.seed)

            if args.gadget_init == 'mask':
                cur_gadget_list = [mask_token_id] * args.num_tokens
            elif args.gadget_init == 'random':
                cur_gadget_list = np.random.choice(token_vocab, size=args.num_tokens).tolist()
            cur_gadget_txt = emb_dict['tokenizer'].decode(cur_gadget_list)

            cur_score = router.calculate_strong_win_rate(cur_gadget_txt)
            print(f"Initial gadget = {cur_gadget_txt} [{cur_score}]")
            cur_loss = - cur_score # we want to maximize the score so if we "minimize" the loss then the loss is just the minus of the score. we can ofcouse use more complex loss functions here, including perpelxity maybe
            gadget_data['gadget_hist'].append(cur_gadget_txt)
            gadget_data['score_hist'].append(cur_score)
            gadget_data['loss_hist'].append(cur_loss)

            es_count = 0
            progress_bar = tqdm.tqdm(range(args.num_iterations), desc=f"Opt gadget {gadget_id}", ncols=100)
            for step in progress_bar:
                # choose cand tokens
                cands = rng.choice(token_vocab, size=args.batch_size, replace=False)
                if 'probs' in args.token_sampling:
                    cands = rng.choice(token_vocab, size=args.batch_size, replace=False, p=token_probs)
                cands = np.sort(cands)
                gadget_data['sampled_token_hist'].append(cands)

                # create cand gadgets by replacing the tokens at the samples location
                cand_gadgets_list = np.array([cur_gadget_list] * args.batch_size)
                loc = rng.choice(args.num_tokens)
                cand_gadgets_list[:, loc] = cands
                gadget_data['rep_loc_hist'].append(loc)
                cand_gadgets_txt = emb_dict['tokenizer'].decode_batch(cand_gadgets_list)

                # compute score for the cand gadgets
                cand_scores = np.array([router.calculate_strong_win_rate(c) for c in cand_gadgets_txt])

                # compute the loss which is just the minus of the score
                loss = - cand_scores

                if loss[np.argmin(loss)] < cur_loss:
                    best_cand_idx = np.argmin(loss)
                    es_count = 0
                    cur_loss = loss[best_cand_idx]
                    cur_gadget_list = cand_gadgets_list[best_cand_idx]
                    gadget_data['idx_chosen_sampled_token_hist'].append(best_cand_idx)
                    gadget_data['val_chosen_sampled_token_hist'].append(cands[best_cand_idx])

                    gadget_data['score_hist'].append(cand_scores[best_cand_idx])
                    gadget_data['gadget_hist'].append(cand_gadgets_txt[best_cand_idx])

                else:
                    #no update in this iter
                    gadget_data['idx_chosen_sampled_token_hist'].append(-1)
                    gadget_data['val_chosen_sampled_token_hist'].append(-1)
                    gadget_data['score_hist'].append(gadget_data['score_hist'][-1])
                    es_count += 1

                gadget_data['loss_hist'].append(cur_loss)
                if step % 1 == 0:
                    progress_bar.set_postfix({'Score': f"{gadget_data['score_hist'][-1]:.4f}"})

                if es_count == args.early_stop:
                    # if we early stopped, fill in the rest of the things so that the array's lengths would match across sampels and it'd be easier to plot
                    gadget_data['score_hist'] += [gadget_data['score_hist'][-1]] * (args.num_iterations - step - 1)
                    gadget_data['loss_hist'] += [gadget_data['loss_hist'][-1]] * (args.num_iterations - step - 1)
                    break

            with open(gadget_path, 'wb') as f:
                pickle.dump(gadget_data, f)
        # print(f"gadget: {gadget_data['gadget_hist'][-1]} [score={gadget_data['score_hist'][-1]}]\n")


    ####################################################################################################################
    ############### Load data and threshold ###############

    questions, q_ids, answers, answers_txt, format_prefix = utils.get_benchmark_questions_and_ids(benchmark_name=args.benchmark,
                                                                                                  num_samples=args.num_samples,
                                                                                                  seed=args.seed)

    threshold = utils.get_calibrated_threshold(benchmark_name=args.benchmark,
                                               router_name=args.router,
                                               router=router,
                                               strong_model_pct=0.5)

    print(f"Threshold = {threshold}")
    ####################################################################################################################
    ############### Evaluate clean performance ###############

    print("-" * 100 + f"\nEvaluate clean {args.num_samples} queries")

    # get clean routing scores
    score_dir = f"./data/{args.benchmark}/clean_scores/"
    os.makedirs(score_dir, exist_ok=True)
    score_path = os.path.join(score_dir, f"{args.router}_{len(questions)}_seed{args.seed}.pkl")
    if os.path.exists(score_path):
        with open(score_path, 'rb') as f:
            clean_routing_scores = pickle.load(f)['routing_scores']
    else:
        clean_routing_scores = [router.calculate_strong_win_rate(cur_query_txt) for cur_query_txt in questions]
        with open(score_path, 'wb') as f:
            pickle.dump({'routing_scores': clean_routing_scores}, f)


    # get clean model responses for both models
    details_str = "clean_responses"

    clean_strong_responses = utils.get_model_responses(args=args,
                                                       full_modelname=routed_pair.strong,
                                                       short_modelname=routed_pair.strong.split('/')[-1],
                                                       quantize=args.quantize_strong,
                                                       benchmark_name=args.benchmark,
                                                       details_str=details_str,
                                                       questions=questions)
    clean_weak_responses = utils.get_model_responses(args=args,
                                                       full_modelname=routed_pair.weak,
                                                       short_modelname=routed_pair.weak.split('/')[-1],
                                                       quantize=args.quantize_weak,
                                                       benchmark_name=args.benchmark,
                                                       details_str=details_str,
                                                       questions=questions)

    # get benchmark specific scores for clean model responses for both models
    clean_strong_bench_scores = utils.get_benchmark_specific_scores(short_modelname=routed_pair.strong.split('/')[-1],
                                                              quantize=args.quantize_strong,
                                                              benchmark_name=args.benchmark,
                                                              details_str=details_str,
                                                              questions=questions,
                                                              answers=answers,
                                                              responses=clean_strong_responses,
                                                              judge_model=args.judge_model,
                                                              seed=args.seed)

    clean_weak_bench_scores = utils.get_benchmark_specific_scores(short_modelname=routed_pair.weak.split('/')[-1],
                                                            quantize=args.quantize_weak,
                                                            benchmark_name=args.benchmark,
                                                            details_str=details_str,
                                                            questions=questions,
                                                            answers=answers,
                                                            responses=clean_weak_responses,
                                                            judge_model=args.judge_model,
                                                            seed=args.seed)

    # get perplexity scores for clean model responses for both models
    clean_strong_ppl_scores = utils.get_ppl_scores(short_modelname=routed_pair.strong.split('/')[-1],
                                                   quantize=args.quantize_strong,
                                                   ppl_model=args.ppl_model,
                                                   benchmark_name=args.benchmark,
                                                   details_str=details_str,
                                                   responses=clean_strong_responses,
                                                   seed=args.seed)

    clean_weak_ppl_scores = utils.get_ppl_scores(short_modelname=routed_pair.weak.split('/')[-1],
                                                   quantize=args.quantize_weak,
                                                   ppl_model=args.ppl_model,
                                                   benchmark_name=args.benchmark,
                                                   details_str=details_str,
                                                   responses=clean_weak_responses,
                                                   seed=args.seed)


    #apply routing on all responses and scores
    routed_clean_responses = np.where(clean_routing_scores >= threshold,
                                      clean_strong_responses, clean_weak_responses)
    routed_clean_bench_scores = np.where(clean_routing_scores >= threshold,
                                         clean_strong_bench_scores, clean_weak_bench_scores)
    routed_clean_ppl_scores = np.where(clean_routing_scores >= threshold,
                                         clean_strong_ppl_scores, clean_weak_ppl_scores)
    # there are a few anomalies with really high perplexity even though they are regular text, so we filter them
    fil_clean_strong_ppl_scores = clean_strong_ppl_scores[clean_strong_ppl_scores < 100]
    fil_clean_weak_ppl_scores = clean_weak_ppl_scores[clean_strong_ppl_scores < 100]
    fil_routed_clean_ppl_scores = routed_clean_ppl_scores[routed_clean_ppl_scores < 100]
    inds_originally_strong = np.where(clean_routing_scores >= threshold)[0]
    inds_originally_weak = np.where(clean_routing_scores < threshold)[0]




    ####################################################################################################################
    ############### Evaluate confounded performance ###############

    confounded_routing_scores = []
    confounded_num_strong, confounded_num_weak = [], []
    num_s2s, num_s2w, num_w2s, num_w2w = [], [], [], []
    confounded_strong_responses, confounded_weak_responses = [], []
    confounded_strong_bench_scores, confounded_weak_bench_scores, routed_confounded_bench_scores = [], [], []
    confounded_strong_ppl_scores, confounded_weak_ppl_scores, routed_confounded_ppl_scores = [], [], []
    fil_confounded_strong_ppl_scores, fil_confounded_weak_ppl_scores, fil_routed_confounded_ppl_scores = [], [], []

    for gadget_id in range(args.num_attacks):
        gadget_path = os.path.join(gadget_dir, f'gadget_{gadget_id}.pkl')
        with open(gadget_path, 'rb') as f:
            cur_gadget = pickle.load(f)['gadget_hist'][-1]

        confounded_queries = [utils.confound_query(cur_query_txt, cur_gadget, args.concat_method, format_prefix) for cur_query_txt in questions]
        cur_confounded_routing_scores = np.array([router.calculate_strong_win_rate(cur_conf_query) for cur_conf_query in confounded_queries])
        confounded_routing_scores.append(cur_confounded_routing_scores)

        details_str = f'{args.router}_conf_responses/{opt_details}/gadget_{gadget_id}'
        confounded_strong_responses.append(utils.get_model_responses(args=args,
                                                           full_modelname=routed_pair.strong,
                                                           short_modelname=routed_pair.strong.split('/')[-1],
                                                           quantize=args.quantize_strong,
                                                           benchmark_name=args.benchmark,
                                                           details_str=details_str,
                                                           questions=confounded_queries))
        confounded_weak_responses.append(utils.get_model_responses(args=args,
                                                         full_modelname=routed_pair.weak,
                                                         short_modelname=routed_pair.weak.split('/')[-1],
                                                         quantize=args.quantize_weak,
                                                         benchmark_name=args.benchmark,
                                                         details_str=details_str,
                                                         questions=confounded_queries))

        confounded_strong_bench_scores.append(utils.get_benchmark_specific_scores(short_modelname=routed_pair.strong.split('/')[-1],
                                                                        quantize=args.quantize_strong,
                                                                        benchmark_name=args.benchmark,
                                                                        details_str=details_str,
                                                                        questions=confounded_queries,
                                                                        answers=answers,
                                                                        responses=confounded_strong_responses[-1],
                                                                        judge_model=args.judge_model,
                                                                        seed=args.seed))


        confounded_weak_bench_scores.append(utils.get_benchmark_specific_scores(short_modelname=routed_pair.weak.split('/')[-1],
                                                                      quantize=args.quantize_weak,
                                                                      benchmark_name=args.benchmark,
                                                                      details_str=details_str,
                                                                      questions=confounded_queries,
                                                                      answers=answers,
                                                                      responses=confounded_weak_responses[-1],
                                                                      judge_model=args.judge_model,
                                                                      seed=args.seed))



        cur_confounded_strong_ppl_scores = utils.get_ppl_scores(short_modelname=routed_pair.strong.split('/')[-1],
                                                                 quantize=args.quantize_strong,
                                                                 ppl_model=args.ppl_model,
                                                                 benchmark_name=args.benchmark,
                                                                 details_str=details_str,
                                                                 responses=confounded_strong_responses[-1],
                                                                 seed=args.seed)
        confounded_strong_ppl_scores.append(cur_confounded_strong_ppl_scores)

        cur_confounded_weak_ppl_scores = utils.get_ppl_scores(short_modelname=routed_pair.weak.split('/')[-1],
                                                                 quantize=args.quantize_weak,
                                                                 ppl_model=args.ppl_model,
                                                                 benchmark_name=args.benchmark,
                                                                 details_str=details_str,
                                                                 responses=confounded_weak_responses[-1],
                                                                 seed=args.seed)
        confounded_weak_ppl_scores.append(cur_confounded_weak_ppl_scores)



        routed_confounded_bench_scores.append(np.where(cur_confounded_routing_scores >= threshold,
                                             confounded_strong_bench_scores[-1], confounded_weak_bench_scores[-1]))
        cur_routed_confounded_ppl_scores = np.where(cur_confounded_routing_scores >= threshold,
                                                     cur_confounded_strong_ppl_scores, cur_confounded_weak_ppl_scores)
        routed_confounded_ppl_scores.append(cur_routed_confounded_ppl_scores)
        # there are a few anomalies with really high perplexity even though they are regular text, so we filter them
        fil_confounded_strong_ppl_scores.append(np.mean(cur_confounded_strong_ppl_scores[cur_confounded_strong_ppl_scores < 100]))
        fil_confounded_weak_ppl_scores.append(np.mean(cur_confounded_weak_ppl_scores[cur_confounded_weak_ppl_scores < 100]))
        fil_routed_confounded_ppl_scores.append(np.mean(cur_routed_confounded_ppl_scores[cur_routed_confounded_ppl_scores < 100]))
        inds_now_strong = np.where(cur_confounded_routing_scores >= threshold)[0]
        inds_now_weak = np.where(cur_confounded_routing_scores < threshold)[0]

        confounded_num_strong.append(len(inds_now_strong))
        confounded_num_weak.append(len(inds_now_weak))

        num_s2s.append(len(np.where(cur_confounded_routing_scores[inds_originally_strong] >= threshold)[0]))
        num_s2w.append(len(np.where(cur_confounded_routing_scores[inds_originally_strong] < threshold)[0]))

        num_w2s.append(len(np.where(cur_confounded_routing_scores[inds_originally_weak] >= threshold)[0]))
        num_w2w.append(len(np.where(cur_confounded_routing_scores[inds_originally_weak] < threshold)[0]))

        print(f"confounded-{gadget_id}:")
        print(f"bench-spec: strong-score: {np.mean(confounded_strong_bench_scores[-1])}, weak-score: {np.mean(confounded_weak_bench_scores[-1])}, routed-score: {np.mean(routed_confounded_bench_scores[-1])}")
        print(f"ppl: strong-score: {np.mean(confounded_strong_ppl_scores[-1])}, weak-score: {np.mean(confounded_weak_ppl_scores[-1])}, routed-score: {np.mean(routed_confounded_ppl_scores[-1])}")
        print(f"fil-ppl: strong-score: {np.mean(fil_confounded_strong_ppl_scores[-1])}, weak-score: {np.mean(fil_confounded_weak_ppl_scores[-1])}, routed-score: {np.mean(fil_routed_confounded_ppl_scores[-1])}")
        print(f"now strong: {len(inds_now_strong)}, now weak: {len(inds_now_weak)}")

    print("Done collecting all confounded results")

    txt = f"-----Routing results-----\n"
    txt += (f"Clean: {len(inds_originally_strong)/len(questions)} ({len(inds_originally_strong)}/{len(questions)}) routed to the strong model and"
            f" {len(inds_originally_weak)/len(questions)} ({len(inds_originally_weak)}/{len(questions)}) to the weak model.\n")
    txt += (f"Confounded (avg): {np.mean(confounded_num_strong)/len(questions)} ({np.mean(confounded_num_strong)}/{len(questions)}) routed to the strong model and"
          f" {np.mean(confounded_num_weak)/len(questions)} ({np.mean(confounded_num_weak)}/{len(questions)}) to the weak model.\n\n")

    txt += (f"Weak -> Strong (upgrade rate, avg): {np.mean(num_w2s)/len(inds_originally_weak)} ({np.mean(num_w2s)}/{len(inds_originally_weak)})\n")
    txt += (f"Weak -> Weak (avg): {np.mean(num_w2w)/len(inds_originally_weak)} ({np.mean(num_w2w)}/{len(inds_originally_weak)})\n")
    txt += (f"Strong -> Strong (avg): {np.mean(num_s2s)/len(inds_originally_strong)} ({np.mean(num_s2s)}/{len(inds_originally_strong)})\n")
    txt += (f"Strong -> Weak (downgrade rate, avg): {np.mean(num_s2w)/len(inds_originally_strong)} ({np.mean(num_s2w)}/{len(inds_originally_strong)})\n\n")

    txt += (f"-----Benchmark-specific scores results-----\n")
    txt += (f"Clean: {np.mean(routed_clean_bench_scores)} (just strong: {np.mean(clean_strong_bench_scores)}, just weak: {np.mean(clean_weak_bench_scores)})\n")
    txt += (f"Confounded: {np.mean(confounded_strong_bench_scores)} (just strong: {np.mean(routed_confounded_bench_scores)}, just weak: {np.mean(confounded_weak_bench_scores)})\n\n")

    txt += (f"-----PPL scores results (filtered)-----\n")
    txt += (f"Clean: {np.mean(fil_routed_clean_ppl_scores)} (just strong: {np.mean(fil_clean_strong_ppl_scores)}, just weak: {np.mean(fil_clean_weak_ppl_scores)})\n")
    txt += (f"Confounded: {np.mean(fil_routed_confounded_ppl_scores)} (just strong: {np.mean(fil_confounded_strong_ppl_scores)}, just weak: {np.mean(fil_confounded_weak_ppl_scores)})\n")


    print(txt)

    results_path = f"./results/{args.benchmark}/{args.router}/strong_{routed_pair.strong.split('/')[-1]}_q-{args.quantize_strong}_weak_{routed_pair.weak.split('/')[-1]}_q-{args.quantize_weak}"
    os.makedirs(results_path, exist_ok=True)
    results_path = os.path.join(results_path, f"results_{opt_details}_{len(questions)}_seed{args.seed}.txt")
    f = open(results_path, "w")
    f.write(txt)
    f.close()






