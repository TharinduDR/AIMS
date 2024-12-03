from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from huggingface_hub import login
from transformers import logging
import torch
import json
import os
from datasets import Dataset
from datasets import load_dataset
#logging.set_verbosity_debug()

data = Dataset.to_pandas(load_dataset("csebuetnlp/xlsum", "english", split='train'))

test = data.iloc[:10000]
train = data.iloc[10000:]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct",
model_representation = model_name.replace('/', '-')

mdlPipeline = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map='auto',
)

def count_tokens(input_text):
    return len(str(input_text).split())

# Filter the training set for rows where 'summary' has more than 30 tokens
filtered_train = train[train['summary'].apply(count_tokens) > 30]


predictions = []
input_list = test['text'].tolist()
truth_list = test['summary'].tolist()

for index, row in test.iterrows():
    random_samples = filtered_train.sample(n=2)  # Set random_state for reproducibility

    text1, text2 = random_samples['text'].tolist()
    summary1, summary2 = random_samples['summary'].tolist()
    text = row['text']

    systemPrompt = f"""
    Here are two examples of an article and its summary. Using these examples, respond to the next message with a summary of the given article as a short paragraph with TWO sentences at most. Do NOT go over this limit.

    Article 1:
    {text1}

    Summary 1:
    {summary1}

    Article 2:
    {text2}

    Summary 2:
    {summary2}

    USE NO LISTS OR BULLET POINTS.
    """

    messages = [
        {"role": "system", "content": systemPrompt},
        {"role": "user", "content": text},
    ]
    outputs = mdlPipeline(
        messages,
        max_new_tokens=1500,
        temperature=0.1
    )

    summary = outputs[0]["generated_text"][-1]["content"]
    predictions.append(summary)

data = {
    "Input": input_list,
    "Prediction": predictions,
    "Truth": truth_list
}
df = pd.DataFrame(data)

# Save as a TSV file
df.to_csv(os.path.join("outputs_xlsum", model_representation, "predictions.tsv"), sep='\t', encoding='utf-8', index=False)

