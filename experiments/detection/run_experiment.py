import os
import shutil
import torch
import numpy as np
from datasets import Dataset
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from aims.config.model_args import TextClassificationArgs

from aims.text_classification.text_classification_model import TextClassificationModel
from experiments.detection.evaluation import macro_f1, weighted_f1

model_name = "google-bert/bert-large-uncased"
model_type = "bert"

dataset_name = "tharindu/mistralai-Mistral-7B-Instruct-v0.3-incontext-xlsum"

train = Dataset.to_pandas(load_dataset(dataset_name, split='train', force_redownload=True))
test = Dataset.to_pandas(load_dataset(dataset_name, split='test', force_redownload=True))

train = train.sample(frac=1)
test = test.sample(frac=1)

train = train.rename(columns={'summary': 'text', 'AI': 'labels'}).dropna()
test = test.rename(columns={'summary': 'text', 'AI': 'labels'}).dropna()

test_sentences = test['text'].tolist()

macrof1_values = []
weightedf1_values = []


for i in range(5):

    model_args = TextClassificationArgs()
    model_args.num_train_epochs = 5
    model_args.no_save = False
    model_args.fp16 = True
    model_args.learning_rate = 1e-5
    model_args.train_batch_size = 8
    model_args.max_seq_length = 512
    model_args.model_name = model_name
    model_args.model_type = model_type
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_verbose = True
    model_args.evaluate_during_training_steps = 500
    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.overwrite_output_dir = True
    model_args.save_recent_only = True
    model_args.logging_steps = 500
    model_args.manual_seed = 777
    model_args.early_stopping_patience = 10
    model_args.save_steps = 500
    model_args.regression = False

    processed_dataset_name = dataset_name.split("/")[1]

    model_args.output_dir = os.path.join("outputs",  processed_dataset_name)
    model_args.best_model_dir = os.path.join("outputs",  processed_dataset_name, "best_model")
    model_args.cache_dir = os.path.join("cache_dir", processed_dataset_name)

    model_args.wandb_project = "AI Summary Detection"
    model_args.wandb_kwargs = {"name": processed_dataset_name}

    if os.path.exists(model_args.output_dir) and os.path.isdir(model_args.output_dir):
        shutil.rmtree(model_args.output_dir)

    model = TextClassificationModel(model_type, model_name, args=model_args, use_cuda=torch.cuda.is_available())
    temp_train, temp_eval = train_test_split(train, test_size=0.2, random_state=model_args.manual_seed*i)
    model.train_model(temp_train, eval_df=temp_eval, macro_f1=macro_f1, weighted_f1=weighted_f1)
    predictions, raw_outputs = model.predict(test_sentences)

    test['predictions'] = predictions
    macro = macro_f1(test["labels"].tolist(), test["predictions"].tolist())
    weighted = weighted_f1(test["labels"].tolist(), test["predictions"].tolist())

    macrof1_values.append(macro)
    weightedf1_values.append(weighted)


print(macrof1_values)
print(weightedf1_values)

print("Mean Macro F1:", np.mean(np.array(macrof1_values)))
print("STD Macro F1:", np.std(np.array(macrof1_values)))

print("Mean Weighted F1:", np.mean(np.array(weightedf1_values)))
print("STD Weighted F1:", np.std(np.array(weightedf1_values)))




