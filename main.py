import os
import time
import json
import torch
import random
import argparse
import numpy as np
from generator import Generator, OnlineGenerator
from prompt import prompt_template
from extractor import Extractor
from bm25 import TaskSpecificBM25
from datasets import load_test_dataset, load_train_and_valid_dataset, construct_dataset, CodeBlock
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils.eval_metric import compute_metric_stmt
from utils.eval_codereval import eval_codereval
from prettytable import PrettyTable
import copy

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set seed
def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()


# Retrieves code blocks based on different inference types.
def retrieve_codeblocks(args, examples, key_word_list, bm25, dataset_name, is_training=False, inference_type=None):
    """
    Retrieves code blocks based on different inference types.
    :param args: An argument object containing configuration parameters.
    :param examples: Examples used for retrieval.
    :param bm25: An instance of the BM25 model.
    :param dataset_name: The name of the dataset.
    :param is_training: Whether it is in training mode.
    :return: A list of retrieved code blocks.
    """
    if inference_type is None:
        inference_type = args.inference_type
    if inference_type == "baseline":
        return None, [[] for _ in range(len(examples))]

    bm25_topk, unixcoder_topk, context_len = 5, 5, 20
    if inference_type in ["bm25", "unixcoder", "unixcoder_with_rl"]:
        if dataset_name not in bm25:
            bm25[dataset_name] = TaskSpecificBM25(examples, args)

        if inference_type == "unixcoder":
            bm25_topk = 50 
        elif inference_type == "unixcoder_with_rl":
            bm25_topk = args.sample_number * 10 
            unixcoder_topk = args.sample_number 

        # queries = ["\n".join([x for x in example.left_context.split("\n") if x.strip() != ""][-context_len:]) for example in examples]
        # candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in examples], queries, topk=bm25_topk)

    # shape of keyword_lists (num_examples, num_samples, key word list)
    if is_training:    
        keyword = [[[kw for kw in sample] for sample in example_keywords] for example_keywords in key_word_list]
        candidate_codeblocks = bm25[dataset_name].query_batch([x.task_id for x in examples], keyword, topk=5)
        # for index_e, example in enumerate(key_word_list):
        #     for index_s, sample in enumerate(example):
        #         print(f"example:{index_e} sample:{index_s}\nkey word list:{sample}")
        # for index_e, example in enumerate(candidate_codeblocks):
        #     for index_s, sample in enumerate(example):
        #         print(f"example:{index_e} sample:{index_s}\ncode block:{sample}")                  
        return None, candidate_codeblocks
    else :
        # when eval, len(num_samples) == 1
        keyword = [sample[0] for sample in key_word_list]  # one sample
        candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in examples], keyword, topk=5)
        # for index_e, example in enumerate(key_word_list):
        #     for index_s, sample in enumerate(example):
        #         print(f"example:{index_e} sample:{index_s}\nkey word list:{sample}")
        # for index_e, example in enumerate(candidate_codeblocks):
        #     for index_s, sample in enumerate(example):
        #         print(f"example:{index_e} sample:{index_s}\ncode block:{sample}")      
        return None, candidate_codeblocks

    raise ValueError("Unsupported inference type: {}".format(args.inference_type))


class ExtractorTrainDataset(Dataset):
    def __init__(self, examples, key_word_lists, tokenizer, args):
        self.examples = examples
        self.key_word_lists = key_word_lists
        self.tokenizer = tokenizer
        self.args = args
        
        # 计算模板长度
        template = prompt_template.format(code="")
        self.template_tokens = self.tokenizer.encode(template, add_special_tokens=False)
        self.template_length = len(self.template_tokens)
        
        # 预留空间给关键词生成（假设最大关键词长度为64）
        self.max_code_length = args.extractor_max_context_length - self.template_length - args.extractor_max_generation_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        keywords = self.key_word_lists[idx]
        
        # 处理代码
        context_tokens = self.tokenizer.encode(
            example.left_context, 
            add_special_tokens=False
        )
        if len(context_tokens) > self.max_code_length:
            context_tokens = context_tokens[-(self.max_code_length):]
        truncated_context = self.tokenizer.decode(context_tokens)
        
        # 处理prompt和keyword
        prompt = prompt_template.format(code=truncated_context)
        prompt_tokens = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
            truncation=False
        )
        target_text = ", ".join(keywords)
        target_tokens = self.tokenizer.encode(
            target_text,
            add_special_tokens=False,
            truncation=False
        )[:self.args.extractor_max_generation_length]

        # 手动拼接
        total_max_length = (
            self.args.extractor_max_context_length 
            + self.args.extractor_max_generation_length
        )
        input_tokens = prompt_tokens + target_tokens
        labels = [-100]*len(prompt_tokens) + target_tokens
        if len(input_tokens) < total_max_length:
            # 右侧填充（保持prompt在前）
            pad_length = total_max_length - len(input_tokens)
            input_tokens += [self.tokenizer.pad_token_id] * pad_length
            labels += [-100] * pad_length
        else:
            # 严格截断（优先保留prompt）
            keep_length = self.args.extractor_max_context_length
            input_tokens = input_tokens[:keep_length] + target_tokens[:self.args.extractor_max_generation_length]
            labels = [-100]*keep_length + target_tokens[:self.args.extractor_max_generation_length]
        
        input_ids = torch.tensor(input_tokens)
        labels = torch.tensor(labels)
        # if idx % 5 == 0:
        #     print("\n===== DEBUG [样本{}] =====".format(idx))
        #     # 验证prompt部分
        #     prompt_part = self.tokenizer.decode(
        #         input_ids[:len(prompt_tokens)], 
        #         skip_special_tokens=True
        #     )
        #     print("[Prompt]:\n", prompt_part.replace("\\n", "\n"))
        #     # 验证目标部分
        #     target_mask = (labels != -100)
        #     if target_mask.any():
        #         target_start = target_mask.nonzero()[0].item()
        #         decoded_target = self.tokenizer.decode(
        #             input_ids[target_start:], 
        #             skip_special_tokens=True
        #         )
        #         print("[Decoded Target]:", decoded_target)
        #         print("[Expected Target]:", target_text)
        #     else:
        #         print("[ERROR] 无有效目标区域!")
            
        #     # Token级别验证
        #     print("\n[Token对齐检查]")
        #     print("Prompt长度:", len(prompt_tokens))
        #     print("Target长度:", len(target_tokens))
        #     print("总长度:", len(input_tokens))
        #     print("标签掩码示例:", labels[:len(prompt_tokens)+3].tolist(), "...")

        return {
            "input_ids": input_ids,
            "attention_mask": (input_ids != self.tokenizer.pad_token_id).int(),
            "labels": labels
        }

def _postprocess_keywords(raw_text):
    keywords = raw_text.strip().split(',')
    return list(dict.fromkeys([k.strip().lower() for k in keywords if k.strip()]))

def run(args):
    cceval_python_examples = load_test_dataset(args, "cceval", "python")
    cceval_java_examples = load_test_dataset(args, "cceval", "java")
    # codereval_python_examples = load_test_dataset(args, "codereval", "python")
    # codereval_java_examples = load_test_dataset(args, "codereval", "java")
    repoeval_line_examples = load_test_dataset(args, "repoeval", "line_level")
    repoeval_api_examples = load_test_dataset(args, "repoeval", "api_level")
    # repoeval_func_examples = load_test_dataset(args, "repoeval", "func_level")

    training_raw_data, eval_raw_data = load_train_and_valid_dataset()
    eval_all_examples = construct_dataset(eval_raw_data, 100 if args.debug else 1000)

    all_eval_examples = {
        "github_eval": eval_all_examples,
        "cceval_python": cceval_python_examples,
        "cceval_java": cceval_java_examples,
        # "codereval_python": codereval_python_examples,
        # "codereval_java": codereval_java_examples,
        "repoeval_line": repoeval_line_examples,
        "repoeval_api": repoeval_api_examples,
        # "repoeval_func": repoeval_func_examples,
    }


    global generator
    generator = Generator(args)
    # online_generator = OnlineGenerator(args)
    # from extractor_eval import Extractor as Extractor_eval
    # extractor = Extractor_eval(args)
    extractor = Extractor(args)

    # if args.enable_repocoder:
    #     args_RLCoder = copy.deepcopy(args)
    #     args_RLCoder.retriever_model_path = args.rlcoder_model_path
    #     global retriever_RLCoder
    #     retriever_RLCoder = Retriever(args_RLCoder)
    

    if not args.enable_forward_generation:
        args.forward_generation_times = 1
    else:
        if args.forward_generation_times is None:
            args.forward_generation_times = 4

    bm25 = {}
    
    if args.eval:
        table = PrettyTable()
        table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        codereval_table = PrettyTable()
        codereval_table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
        
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            print("Evaluating on {} dataset".format(name))
            
            temp_examples = copy.deepcopy(examples)
            temp_generations = []
                
            for _ in range(args.forward_generation_times):
                # _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                # for i in range(len(retrieved_codeblocks)):
                #     for j in range(len(retrieved_codeblocks[i])):
                #         print('#', retrieved_codeblocks[i][j].file_path)
                #         print(retrieved_codeblocks[i][j].code_content)
                key_word_list = extractor.generate(temp_examples,args.extractor_max_generation_length, num_return_sequences=1)
                process_key_word_list = [[_postprocess_keywords(text) for text in sample] for sample in key_word_list]
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, process_key_word_list, bm25, name, is_training=False)
                losses = generator.evaluate(examples, retrieved_codeblocks)


                results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if not os.path.exists(f"{args.output_dir}/{name}"):
                    os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
                with open(f"{args.output_dir}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")


                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, temp_generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in temp_generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(temp_generations,targets)])/len(temp_generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                # elif name == "repoeval_func":
                #     results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/func_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
            

            if 'codereval' in name:
                codereval_table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(table)
            print(codereval_table)
        
    else:
        print("data_per_epoch:{}, batch_size:{}, sample_number:{}, epoch:{}, inner_epoch:{}, lr:{}".format(args.data_per_epoch, args.batch_size,args.sample_number,args.epoch,args.inner_epoch,args.lr))
        # optimizer = AdamW(extractor.model.parameters(), lr=args.lr, eps=1e-8)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.2, num_training_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch)
        optimizer = AdamW(
                    extractor.model.parameters(),
                    lr=args.lr, 
                    eps=1e-8,
                    weight_decay=0.01  
                    )
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=int(args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.1),  
                    num_training_steps=args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch,
                    num_cycles=0.5  
                    )
        evaluate_table = {}
        for name, examples in all_eval_examples.items():
            evaluate_table[name] = PrettyTable()
            if 'codereval' in name:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
            else:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]

        training_table = PrettyTable()
        training_table.field_names = ["Epoch", "Dataset", "Total Samples", "Training Loss", "Time (sec)"]


        extractor.model.eval()
        for name, examples in all_eval_examples.items():
            # examples = examples[:10]
            
            start_time = time.time()
            temp_examples = copy.deepcopy(examples)
            temp_generations = []

                
            for _ in range(args.forward_generation_times):
                # _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                # losses = generator.evaluate(examples, retrieved_codeblocks)
                key_word_list = extractor.generate(temp_examples,args.extractor_max_generation_length, num_return_sequences=1)
                process_key_word_list = [[_postprocess_keywords(text) for text in sample] for sample in key_word_list]
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, process_key_word_list, bm25, name, is_training=False)
                losses = generator.evaluate(examples, retrieved_codeblocks)

                results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if os.path.exists(f"{args.output_dir}/result_init/{name}") is False:
                    os.makedirs(f"{args.output_dir}/result_init/{name}", exist_ok=True)
                with open(f"{args.output_dir}/result_init/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")

                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations, targets)])/len(generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                

            if 'codereval' in name:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(evaluate_table[name])


        for epoch in range(args.epoch):
            print("=" * 40 + "Epoch:{}".format(epoch) + "=" * 40)
            extractor.model.eval()
            start_time = time.time()
            results = {}
            results["Epoch"] = epoch


            training_examples = construct_dataset(training_raw_data, 100 if args.debug else args.data_per_epoch)

            key_word_list = extractor.generate(training_examples, args.generator_max_generation_length, num_return_sequences=4)
            process_key_word_list = [[_postprocess_keywords(text) for text in sample] for sample in key_word_list]
            _, retrieved_codeblocks = retrieve_codeblocks(args, training_examples, process_key_word_list, bm25, name, is_training=True)
            losses, best_retrieve_codeblocks, best_key_word_lists = generator.evaluate_samples(training_examples, retrieved_codeblocks, process_key_word_list)

            results["Total Samples"] = len(training_examples)

            extractor.model.train()
            train_dataset = ExtractorTrainDataset(
                examples=training_examples,
                key_word_lists=best_key_word_lists,  
                tokenizer=extractor.tokenizer,
                args=args
            )

            dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda batch: {
                    'input_ids': torch.stack([x['input_ids'] for x in batch]),
                    'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
                    'labels': torch.stack([x['labels'] for x in batch])
                }
            )

            total_loss = 0.0
            num_batch = 0
            for batch in dataloader:
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')  
                labels = batch['labels'].to('cuda')        
                # 前向传播
                outputs = extractor.model(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                # 反向传播
                loss = outputs.loss
                loss.backward()

                total_norm = torch.nn.utils.clip_grad_norm_(extractor.model.parameters(), 1.0)
                # print(f"Gradient norm: {total_norm:.4f}")
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                num_batch += 1
            scheduler.step()
            # 记录训练损失
            results["Training Loss"] = total_loss / num_batch



            if args.enable_sft:
                extractor.model.eval()
                for name, examples in all_eval_examples.items():
                    # examples = examples[:10]
                    
                    start_time = time.time()
                    temp_examples = copy.deepcopy(examples)
                    temp_generations = []

                        
                    for _ in range(args.forward_generation_times):
                        # _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                        # losses = generator.evaluate(examples, retrieved_codeblocks)
                        key_word_list = extractor.generate(temp_examples,args.extractor_max_generation_length, num_return_sequences=1)
                        process_key_word_list = [[_postprocess_keywords(text) for text in sample] for sample in key_word_list]
                        _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, process_key_word_list, bm25, name, is_training=False)
                        losses = generator.evaluate(examples, retrieved_codeblocks)

                        results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                        if args.enable_generation:
                            generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                            if not temp_generations:
                                temp_generations = generations
                            else:
                                temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                            for i in range(len(temp_examples)):
                                temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                                
                    if args.enable_generation:
                        if os.path.exists(f"{args.output_dir}/result_{inner_epoch}/{name}") is False:
                            os.makedirs(f"{args.output_dir}/result_{inner_epoch}/{name}", exist_ok=True)
                        with open(f"{args.output_dir}/result_{inner_epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                            for example, generation in zip(examples, temp_generations):
                                f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                        if name == "cceval_python":
                            results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                        elif name == "cceval_java":
                            results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                        elif name == "github_eval":
                            targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                            results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                        elif name == "codereval_python":
                            results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                        elif name == "codereval_java":
                            results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                        elif name == "repoeval_line":
                            results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                        elif name == "repoeval_api":
                            results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                    if 'codereval' in name:
                        evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                    else:
                        evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                    print(evaluate_table[name])
                extractor.model.module.base_model.save_pretrained(f"{args.output_dir}/extractor/result_{epoch}")

            results["Time (sec)"] = round(time.time() - start_time, 1)
            training_table.add_row([results["Epoch"], "github_training_{}".format(epoch), results["Total Samples"], results["Training Loss"], results["Time (sec)"]])
            print(training_table)

            
            extractor.model.eval()
            for name, examples in all_eval_examples.items():
                # examples = examples[:10]
                
                start_time = time.time()
                temp_examples = copy.deepcopy(examples)
                temp_generations = []
                    
                for _ in range(args.forward_generation_times):
                    # _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name) 
                    # losses = generator.evaluate(examples, retrieved_codeblocks)
                    key_word_list = extractor.generate(temp_examples,args.extractor_max_generation_length, num_return_sequences=1)
                    process_key_word_list = [[_postprocess_keywords(text) for text in sample] for sample in key_word_list]
                    _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, process_key_word_list, bm25, name, is_training=False)
                    losses = generator.evaluate(examples, retrieved_codeblocks)
                    
                    results = {"em": "-","es": "-","id_em": "-","id_f1": "-"}
                    if args.enable_generation:
                        generations = generator.generate(temp_examples, retrieved_codeblocks, args.generator_max_generation_length)

                        if not temp_generations:
                            temp_generations = generations
                        else:
                            temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                        for i in range(len(temp_examples)):
                            temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                            
                if args.enable_generation:
                    if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
                        os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
                    with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                        for example, generation in zip(examples, temp_generations):
                            f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                    if name == "cceval_python":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "cceval_java":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                    elif name == "github_eval":
                        targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                        results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                    elif name == "codereval_python":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                    elif name == "codereval_java":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                    elif name == "repoeval_line":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "repoeval_api":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                if 'codereval' in name:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                else:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

                print(evaluate_table[name])

            extractor.model.module.base_model.save_pretrained(f"{args.output_dir}/extractor/result_{epoch}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--generator_model_path", default="deepseek-ai/deepseek-coder-1.3b-base", type=str, help="Generator model path")
    parser.add_argument("--generator_batch_size_per_gpu", default=32, type=int, help="Generator batch size per GPU")
    parser.add_argument("--generator_max_crossfile_length", default=512, type=int, help="Maximum cross-file length for the generator")
    parser.add_argument("--generator_max_context_length", default=1024, type=int, help="Maximum context length for the generator")
    parser.add_argument("--generator_max_generation_length", default=64, type=int, help="Maximum generation length for the generator")
    parser.add_argument("--disable_generator", action="store_true", help="Disable the generator")

    parser.add_argument("--extractor_model_path", default="Qwen/Qwen2.5-1.5B-Instruct", type=str, help="Extractor model path")
    parser.add_argument("--extractor_batch_size_per_gpu", default=32, type=int, help="Extractor batch size per GPU")
    parser.add_argument("--extractor_max_context_length", default=1024, type=int, help="Maximum context length for the generator")
    parser.add_argument("--extractor_max_generation_length", default=64, type=int, help="Maximum generation length for the generator")
    parser.add_argument("--disable_extractor", action="store_true", help="Disable the generator")

    # parser.add_argument("--retriever_model_path", default="microsoft/unixcoder-base", type=str, help="Retriever model path")
    parser.add_argument("--retriever_batch_size_per_gpu", default=64, type=int, help="Retriever batch size per GPU")
    parser.add_argument("--disable_retriever", action="store_true", help="Disable the retriever")
    parser.add_argument("--retriever_query_context_length", default=256, type=int, help="Retriever query context length")
    parser.add_argument("--retriever_candidate_context_length", default=512, type=int, help="Retriever candidate context length")

    parser.add_argument("--inference_type", default="baseline", type=str, help="Inference type")
    parser.add_argument("--output_dir", default="results/baseline", type=str, help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation")
    parser.add_argument("--enable_tqdm", action="store_true", help="Enable progress bar")
    parser.add_argument("--enable_generation", action="store_true", help="Enable generation")
    parser.add_argument("--debug", action="store_true", help="Debug mode, use a small dataset")

    parser.add_argument("--num_workers", default=14, type=int, help="Number of CPU cores")
    parser.add_argument("--weighted_keywords", action="store_true", help="Weight keywords when calculating loss during training")
    parser.add_argument("--enable_fixed_block", action="store_true", help="Use fixed length blocks when building candidates")
    parser.add_argument("--enable_sft", action="store_true", help="Train using supervised learning methods")
    parser.add_argument("--disable_stop_block", action="store_true", help="Disable the stop block")

    parser.add_argument("--enable_repocoder", action="store_true", help="Use the repocoder method during generation")
    parser.add_argument("--rlcoder_model_path", default="microsoft/unixcoder-base", type=str, help="Stage 1 model for repocoder")

    parser.add_argument("--do_codereval", action="store_true", help="Execute codereval evaluation in docker")
    parser.add_argument("--enable_forward_generation", action="store_true", help="Use progressive generation methods during inference")
    parser.add_argument("--forward_generation_times", default=4, type=int, help="Number of times for progressive generation")

    parser.add_argument("--epoch", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--inner_epoch", default=1, type=int, help="Number of inner training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--sample_number", default=10, type=int, help="Number of samples")
    parser.add_argument("--data_per_epoch", default=2000, type=int, help="Amount of data per epoch")
    parser.add_argument("--lr", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--lora_checkpoint_path", default="result_train/deepseekcoder_1.3b_qwen2.5_1.5b/extractor/result_0", type=str)


    print("Number of GPUs:", torch.cuda.device_count())

    args = parser.parse_args()
    args.generator_batch_size = args.generator_batch_size_per_gpu * torch.cuda.device_count()
    args.extractor_batch_size = args.extractor_batch_size_per_gpu * torch.cuda.device_count()

    run(args)
