from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer,TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml

"""Import PolymerSmilesTokenizer from PolymerSmilesTokenization.py"""
#PolymerSmilesTokenizer是一个专门用于处理聚合物SMILES字符串的定制化tokenizer。
from PolymerSmilesTokenization import PolymerSmilesTokenizer

"""Import LoadPretrainData"""
#LoadPretrainData函数负责加载和预处理用于预训练的数据。
from dataset import LoadPretrainData

"""Device"""
#如果有GPU可用,它将用于训练模型,否则将使用CPU。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available() #checking if CUDA + Colab GPU works

"""train-validation split"""
"""
训练-验证分割
这个函数以文件路径为输入,使用scikit-learn中的train_test_split将数据集分割为训练集和验证集。
该函数返回训练数据和验证数据。
"""
def split(file_path):
    dataset = pd.read_csv(file_path, header=None).values
    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=1)
    return train_data, valid_data


def main(pretrain_config):
    """Use Roberta configuration"""
    """
    预训练 RoBERTa 模型的主函数
    
    Args:
        pretrain_config (dict): 预训练配置参数,从YAML文件中读取

    Returns:
        None
    """

    # 1. 初始化 RobertaConfig 对象
    config = RobertaConfig(
        #设置了模型使用的词汇表大小为50,265,这是RoBERTa模型预定义的词汇表大小
        vocab_size=50265,
        #设置了模型能够处理的最大位置编码长度
        max_position_embeddings=pretrain_config['max_position_embeddings'],
        #设置了Transformer模型中多头注意力机制的注意力头数量
        num_attention_heads=pretrain_config['num_attention_heads'],
        #设置了RoBERTa模型中Transformer编码器层的数量
        num_hidden_layers=pretrain_config['num_hidden_layers'],
        #用来编码输入序列中不同类型的元素,比如区分问题部分和答案部分
        type_vocab_size=1,
        #设置了Transformer模型隐藏层的dropout概率
        hidden_dropout_prob=pretrain_config['hidden_dropout_prob'],
        #设置了Transformer模型注意力机制中注意力概率的dropout概率
        attention_probs_dropout_prob=pretrain_config['attention_probs_dropout_prob'],
    )

    """Set tokenizer"""
    # 2. 设置 Tokenizer
    '''
    通过使用自定义的 PolymerSmilesTokenizer,确保 Tokenizer 能够更好地处理与化学结构相关的输入数据。
    通常,在处理特定领域的数据时,使用自定义的 Tokenizer 可以带来更好的性能。
    在后续的模型训练和推理过程中,这个 Tokenizer 实例将被用于将原始文本转换为模型可以接受的数字序列输入。
    '''
    #tokenizer = RobertaTokenizer.from_pretrained("roberta-base",max_len=512)
    #从预训练的 RoBERTa 模型中加载词汇表和 tokenizer 配置,以确保与预训练模型的兼容性
    tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=pretrain_config['blocksize'])

    """Construct MLM model"""
    # 3. 构建 RobertaForMaskedLM 模型并将其移动到设备上
    model = RobertaForMaskedLM(config=config).to(device)

    """Load Data"""
    # 4. 加载训练集和验证集
    train_data, valid_data = split(pretrain_config['file_path'])
    data_train = LoadPretrainData(tokenizer=tokenizer, dataset=train_data, blocksize=pretrain_config['blocksize'])
    data_valid = LoadPretrainData(tokenizer=tokenizer, dataset=valid_data, blocksize=pretrain_config['blocksize'])

    """Set DataCollator"""
    # 5. 设置数据收集器
    #DataCollatorForLanguageModeling 实例,它是一个数据预处理器,用于准备输入数据以进行掩码语言模型(Masked Language Modeling, MLM)训练
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=pretrain_config['mlm_probability']
    )

    """Training arguments"""
    # 6. 配置训练参数
    training_args = TrainingArguments(
        output_dir=pretrain_config['save_path'],
        overwrite_output_dir=pretrain_config['overwrite_output_dir'],
        num_train_epochs=pretrain_config['epochs'],
        per_device_train_batch_size=pretrain_config['batch_size'],
        per_device_eval_batch_size=pretrain_config['batch_size'],
        save_strategy=pretrain_config['save_strategy'],
        save_total_limit=pretrain_config['save_total_limit'],
        fp16=pretrain_config['fp16'],
        logging_strategy=pretrain_config['logging_strategy'],
        evaluation_strategy=pretrain_config['evaluation_strategy'],
        learning_rate=pretrain_config['lr_rate'],
        lr_scheduler_type=pretrain_config['scheduler_type'],
        weight_decay=pretrain_config['weight_decay'],
        warmup_ratio=pretrain_config['warmup_ratio'],
        report_to=pretrain_config['report_to'],
        dataloader_num_workers=pretrain_config['dataloader_num_workers'],
        sharded_ddp=pretrain_config['sharded_ddp'],
    )

    """Set Trainer"""
    # 7. 初始化 Trainer 对象
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=data_train,
        eval_dataset=data_valid
    )

    """
    # 8. 设置 TensorBoard
    writer = SummaryWriter(log_dir=training_args.logging_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', training_args.logging_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    """
    

    """Train and save model"""
    # 9. 训练模型并保存
    #torch.cuda.empty_cache()
    #trainer.train()
    trainer.train(resume_from_checkpoint=pretrain_config['load_checkpoint'])
    trainer.save_model(pretrain_config["save_path"])

if __name__ == "__main__":

    #这行代码将YAML配置文件中的内容解析为一个Python字典,并赋值给变量pretrain_config
    pretrain_config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    """Run the main function"""
    main(pretrain_config)