import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
from utils.eval_utils import is_identifier
import re
from vllm import LLM, SamplingParams
class CustomDataset(Dataset):
    """
    A dataset class for code generation.

    Args:
        args: Configuration parameters.
        tokenizer: Tokenizer.
        examples: A collection of examples.
        retrieved_codeblocks: Retrieved code blocks.
    """
    def __init__(self, args, tokenizer, examples, retrieved_codeblocks, generation=False):
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        self.retrieved_codeblocks = retrieved_codeblocks
        self.generation = generation

    def __len__(self):
        return len(self.examples)

    def construct_prompts(self,example,retrieved_codeblocks):
        filter_codeblocks = []
        for x in retrieved_codeblocks:
            # Keep only the blocks before the empty block
            if x.file_path != "":
                filter_codeblocks.append(x)
            else:
                break
        crossfile_context = "\n\n".join([str(retrieved_codeblock) for retrieved_codeblock in filter_codeblocks])
        crossfile_context = self.tokenizer.encode(crossfile_context[:self.args.generator_max_crossfile_length*10], add_special_tokens=False)[:self.args.generator_max_crossfile_length]
        path_context = f"\n\n# file path: {example.file_path}\n\n"
        path_context = self.tokenizer.encode(path_context, add_special_tokens=False)
        allowed_prompt_length = self.args.generator_max_context_length - (len(crossfile_context)+len(path_context)+10)
        infile_context = self.tokenizer.encode(example.left_context, add_special_tokens=False)[-allowed_prompt_length:]

        prompt = self.tokenizer.decode(crossfile_context + path_context + infile_context)
        return prompt

    def __getitem__(self, idx):
        example = self.examples[idx]
        retrieved_codeblocks = self.retrieved_codeblocks[idx]
        prompt = self.construct_prompts(example,retrieved_codeblocks)
        
        prompt_ids = self.tokenizer.encode(prompt)[-self.args.generator_max_context_length:]
        if self.generation:
             padding_length = self.args.generator_max_context_length - len(prompt_ids)
             input_ids = [self.tokenizer.pad_token_id] * padding_length + prompt_ids
             return torch.tensor(input_ids)

        target_ids = self.tokenizer.encode(example.target_code, add_special_tokens=False)[:self.args.generator_max_generation_length]

        input_ids = prompt_ids + target_ids
        labels = [-100 for _ in prompt_ids] + target_ids

        padding_length = self.args.generator_max_context_length + self.args.generator_max_generation_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        labels = [-100] * padding_length + labels 

        return torch.tensor(input_ids), torch.tensor(labels)

class Model(nn.Module):
    def __init__(self, generator_model_path, tokenizer, max_generation_length=64):
        super(Model, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(generator_model_path, torch_dtype=torch.float16)
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length

    def forward(self, inputs=None, labels=None, lang='python', weighted_keywords=False):
        """
        Forward propagation method for calculating loss.
        :param inputs: Input data.
        :param labels: Label data.
        :return: The average loss per sample.
        """
        if labels is not None:
            logits = self.base_model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id))[0]
            logits = logits[:, :-1]
            labels = labels[:, 1:]
            
            label_tokens = [self.tokenizer.convert_ids_to_tokens(id.item()) if id != -100 else '<pad>' for id in labels.reshape(-1)]

            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction='none')

            if weighted_keywords:
                id_weight = 3
                first_token_weight = 5 

                weights = torch.tensor([
                    first_token_weight if i < 1 else id_weight if is_identifier(token, lang) and any(c.isalpha() or c.isdigit() or c == '_' for c in token) else 1
                    for i, token in enumerate(label_tokens)
                ], dtype=torch.float).cuda()

                loss = loss * weights  

            loss_per_label = loss.reshape(labels.size(0), -1).sum(dim=1) / labels.ne(-100).sum(dim=1)
            
            return loss_per_label
        else:
            generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1)+self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id)
            return generated_ids[:, inputs.size(1):]

class Generator:
    """
    Code generator class. Only using it for calculating PPL as reward in RL. 

    Args:
        args: Configuration parameters.
    """
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not args.disable_generator:
            self.model = Model(args.generator_model_path, self.tokenizer)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()

        self.args = args

    def evaluate(self, examples, retrieved_codeblocks):
        """
        Evaluates the generated code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.

        Returns:
            A list of loss values.
        """
        losses = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm)
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = [x.cuda() for x in batch]
                loss_per_label = self.model(inputs, labels, lang=examples[0].language, weighted_keywords=self.args.weighted_keywords)
                losses.extend(loss_per_label.tolist())
                current_ppl = np.exp(np.mean(losses))
                pbar.set_description(f"Loss/PPL: {np.mean(losses):.3f}/{current_ppl:.3f}")
                
        return losses
    
    def generate(self, examples, retrieved_codeblocks, max_generation_length):
        """
        Generates code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.
            max_generation_length: Maximum length of generation.

        Returns:
            A list of generated codes.
        """
        generated_codes = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks,generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
        if hasattr(self.model, "module"):
            self.model.module.max_generation_length = max_generation_length
        else:
            self.model.max_generation_length = max_generation_length

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
        with torch.no_grad():
            for batch in pbar:
                generated_codes.append(self.model(batch.cuda()))
        generated_codes = torch.cat(generated_codes,0)
        return  [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_codes]
        
    def evaluate_samples(self, examples, retrieved_codeblocks_samples, key_word_list_samples):
        num_samples = len(retrieved_codeblocks_samples[0])
        num_examples = len(examples)
        all_losses = []

        for sample_idx in range(num_samples):
            current_retrieved_codeblocks = [sample[sample_idx] for sample in retrieved_codeblocks_samples]
            
            dataset = CustomDataset(self.args, self.tokenizer, examples, current_retrieved_codeblocks)
            sampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
            
            sample_losses = []
            with torch.no_grad():
                for batch in dataloader:
                    inputs, labels = [x.cuda() for x in batch]
                    loss_per_label = self.model(inputs, labels, lang=examples[0].language, weighted_keywords=self.args.weighted_keywords)
                    sample_losses.extend(loss_per_label.tolist())
            
            all_losses.append(sample_losses)

        # Transpose all_losses，turn it to [num_examples][num_samples]
        all_losses = list(map(list, zip(*all_losses)))

        best_losses = []
        best_retrieve_codeblocks = []
        best_key_word_lists = []

        for idx, example_losses in enumerate(all_losses):
            best_sample_idx = min(range(num_samples), key=lambda i: np.exp(example_losses[i]))
            
            best_losses.append(example_losses[best_sample_idx])
            best_retrieve_codeblocks.append(retrieved_codeblocks_samples[idx][best_sample_idx])
            best_key_word_lists.append(key_word_list_samples[idx][best_sample_idx])

        return best_losses, best_retrieve_codeblocks, best_key_word_lists


class OnlineGenerator:
    """
    Code generator class. Only using it for inference by LLM which deployed via vllm. 

    Args:
        args: Configuration parameters.
    """
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.client = OpenAI(
            base_url="http://localhost:8000/v1",  
            api_key="EMPTY"        
        )

        self.args = args
    
    def generate(self, examples, retrieved_codeblocks, max_generation_length):
        """
        Generates code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.
            max_generation_length: Maximum length of generation.

        Returns:
            A list of generated codes.
        """
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, 
                            batch_size=10000000,  
                            num_workers=self.args.num_workers)

        with torch.no_grad():
            for batch in dataloader:
                prompts = [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch]
                
                outputs = self.client.completions.create(
                    model=self.args.generator_model_path,
                    prompt=prompts,
                    temperature=0, 
                    top_p=1,
                    max_tokens=max_generation_length,
                    timeout=7200 # longer timeout setting to avoid requesting timeout, because the large numb of prompts.
                )

        return  [output.text for output in outputs.choices]






# class Model(nn.Module):
#     def __init__(self, generator_model_path, tokenizer, max_generation_length=64):
#         super(Model, self).__init__()
#         self.base_model = AutoModelForCausalLM.from_pretrained(generator_model_path, torch_dtype=torch.float16)
#         self.tokenizer = tokenizer
#         self.max_generation_length = max_generation_length

#     def forward(self, inputs=None, labels=None, lang='python', weighted_keywords=False):
#         """
#         Forward propagation method for calculating loss.
#         :param inputs: Input data.
#         :param labels: Label data.
#         :return: The average loss per sample.
#         """
#         if labels is not None:
#             logits = self.base_model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id))[0]
#             logits = logits[:, :-1]
#             labels = labels[:, 1:]
            
#             label_tokens = [self.tokenizer.convert_ids_to_tokens(id.item()) if id != -100 else '<pad>' for id in labels.reshape(-1)]

#             loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction='none')

#             if weighted_keywords:
#                 id_weight = 3
#                 first_token_weight = 5 

#                 weights = torch.tensor([
#                     first_token_weight if i < 1 else id_weight if is_identifier(token, lang) and any(c.isalpha() or c.isdigit() or c == '_' for c in token) else 1
#                     for i, token in enumerate(label_tokens)
#                 ], dtype=torch.float).cuda()

#                 loss = loss * weights  

#             loss_per_label = loss.reshape(labels.size(0), -1).sum(dim=1) / labels.ne(-100).sum(dim=1)
            
#             return loss_per_label
#         else:
#             generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1)+self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id)
#             return generated_ids[:, inputs.size(1):]
       

    
# class Generator:
#     """
#     Code generator class.

#     Args:
#         args: Configuration parameters.
#     """
#     def __init__(self, args):
#         self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
#         self.tokenizer.model_max_length = 1e10
#         if self.tokenizer.pad_token_id == None:
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

#         if not args.disable_generator:
#             self.model = Model(args.generator_model_path, self.tokenizer)
#             self.model = torch.nn.DataParallel(self.model).cuda()
#             self.model.eval()

#         self.args = args

#     def evaluate(self, examples, retrieved_codeblocks):
#         """
#         Evaluates the generated code.

#         Args:
#             examples: A collection of examples.
#             retrieved_codeblocks: Retrieved code blocks.

#         Returns:
#             A list of loss values.
#         """
#         losses = []
#         dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks)
#         sampler = SequentialSampler(dataset)
#         dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)

#         pbar = tqdm(dataloader, disable=not self.args.enable_tqdm)
#         with torch.no_grad():
#             for batch in pbar:
#                 inputs, labels = [x.cuda() for x in batch]
#                 loss_per_label = self.model(inputs, labels, lang=examples[0].language, weighted_keywords=self.args.weighted_keywords)
#                 losses.extend(loss_per_label.tolist())
#                 current_ppl = np.exp(np.mean(losses))
#                 pbar.set_description(f"Loss/PPL: {np.mean(losses):.3f}/{current_ppl:.3f}")

#         return losses
    
#     def generate(self, examples, retrieved_codeblocks, max_generation_length):
#         """
#         Generates code.

#         Args:
#             examples: A collection of examples.
#             retrieved_codeblocks: Retrieved code blocks.
#             max_generation_length: Maximum length of generation.

#         Returns:
#             A list of generated codes.
#         """
#         generated_codes = []
#         dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks,generation=True)
#         sampler = SequentialSampler(dataset)
#         dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
#         if hasattr(self.model, "module"):
#             self.model.module.max_generation_length = max_generation_length
#         else:
#             self.model.max_generation_length = max_generation_length

#         pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
#         with torch.no_grad():
#             for batch in pbar:
#                 generated_codes.append(self.model(batch.cuda()))
#         generated_codes = torch.cat(generated_codes,0)
#         return  [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_codes]



