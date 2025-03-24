import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np 
import re
from prompt import prompt_template
from peft import LoraConfig, get_peft_model, TaskType
class ExtractorDataset(Dataset):
    def __init__(self, examples, tokenizer, args):
        self.examples = examples
        self.tokenizer = tokenizer
        self.args = args
        
        template = prompt_template.format(code="")
        self.template_length = len(tokenizer.encode(template, add_special_tokens=False))
        
        self.max_code_length = args.extractor_max_context_length - self.template_length 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
            example = self.examples[idx]

            full_context = example.left_context
            
            context_tokens = self.tokenizer.encode(
                full_context, 
                add_special_tokens=False
            )
            
            if len(context_tokens) > self.max_code_length:
                context_tokens = context_tokens[-self.max_code_length:]
            truncated_context = self.tokenizer.decode(context_tokens)

            prompt = prompt_template.format(code=truncated_context)
            
            inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.args.extractor_max_context_length,
                truncation=True,
                return_tensors="pt"
            )
            
            return inputs.input_ids.squeeze(0)

class Model(nn.Module):
    def __init__(self, generator_model_path, tokenizer, max_generation_length=128, lora_checkpoint_path=None):  # 新增参数
        super(Model, self).__init__()
        # 加载基础模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            generator_model_path, 
            torch_dtype=torch.float16,
        )
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length

        # 关键修改：加载 LoRA 权重（而不是初始化新配置）
        if lora_checkpoint_path is not None:
            from peft import PeftModel
            self.base_model = PeftModel.from_pretrained(
                self.base_model,
                lora_checkpoint_path,
                torch_dtype=torch.float16
            )
            print(f"### check point:{lora_checkpoint_path}")
            # 可选：合并权重到基础模型（提升推理速度）
            self.base_model = self.base_model.merge_and_unload()  
        else:
            # 如果未提供 checkpoint，回退到初始化新 LoRA（仅用于训练）
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            self.base_model.print_trainable_parameters()

        self.base_model.eval()  # 确保在评估模式

    def forward(self, inputs, attention_mask=None, labels=None,temperature=0.7, top_p=0.95, num_return_sequences=4):
        if labels is not None:
            outputs = self.base_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            return outputs
        else:
            if num_return_sequences == 1:
                generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1)+self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id,
                                                    do_sample=False)
            else :
                generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1)+self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id,
                                                    do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=num_return_sequences)
            return generated_ids[:, inputs.size(1):]

class Extractor:       
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.extractor_model_path, padding_side='left')
        self.tokenizer.model_max_length = int(1e6)
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


        
        self.model = Model(
            generator_model_path=args.extractor_model_path,
            tokenizer=self.tokenizer,
            max_generation_length=args.extractor_max_generation_length,  # 确保传递参数
            lora_checkpoint_path=args.lora_checkpoint_path
        )
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.eval()

        self.args = args
    
    def generate(self, examples, max_generation_length, temperature=0.7, top_p=0.95, num_return_sequences=4):
        # 创建数据集
        dataset = ExtractorDataset(examples=examples,tokenizer=self.tokenizer,args=self.args)
        # 创建DataLoader
        dataloader = DataLoader(dataset,batch_size=self.args.extractor_batch_size,sampler=SequentialSampler(dataset),num_workers=self.args.num_workers)
        
        all_results = []
        current_model = self.model.module if hasattr(self.model, "module") else self.model
        current_model.max_generation_length = max_generation_length
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch.to("cuda")
                
                outputs = current_model(inputs=inputs, temperature=temperature, top_p=top_p, num_return_sequences=num_return_sequences)
                
                decoded_outputs = self.tokenizer.batch_decode(
                    outputs.cpu(),
                    skip_special_tokens=True
                )
                batch_keywords = [
                    self._postprocess_keywords(text) 
                    for text in decoded_outputs
                ]
                
                # 重组结果（考虑num_return_sequences）
                if num_return_sequences > 1:
                    results_per_sample = num_return_sequences
                    for i in range(0, len(batch_keywords), results_per_sample):
                        all_results.append(
                            batch_keywords[i:i+results_per_sample]
                        )
                else:
                    all_results.extend([[kw] for kw in batch_keywords])

        return all_results

    # def _postprocess_keywords(self, raw_text):
    #     keywords = raw_text.strip().split(',')
    #     return list(dict.fromkeys([k.strip().lower() for k in keywords if k.strip()]))