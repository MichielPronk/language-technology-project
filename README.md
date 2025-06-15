# Language Technology Project: Team 3
This reposistory contains the code and evaluation script we used to prompt and evaluate the language model for the project.

### Team Members
Jelmer Top<br>
Michiel Pronk<br>
Roman Terpstra<br>
Sem Huisman

# Usage main.py
The code was written to run in Python 3.11. A requirements.txt is included with the requirements for the project.
The following parameters can be passed along when running the ```main.py``` script:
```
--split         The split to use for prompting. Can be either 'train', 'validation', or 'test'.
--model         The model to use. Can be either 'llama' or 'deepseek'.
--fewshot       Whether to use few-shot prompting or not. Add to use few-shot prompting.
--rag           Whether to use RAG or not. Add to use RAG.
--hf_token      The Hugging Face token to use for the project. Needed for Llama if HF_TOKEN is not set.
--index_range   The range of the split to be be predicted. Needs two integers.
```
### Example usage:
```
python main.py --split test --model deepseek --fewshot --hf_token secret --index_range 1107 1264
```

# Usage evaluate.py
The following parameters can be passed along when running the ```evaluate.py``` script:
```
--filename    The filename of the predictions to evaluate.
```
### Example usage:
```
python evaluate.py --filename predictions.json
```