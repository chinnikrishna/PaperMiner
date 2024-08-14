# PaperMiner
Extracts paper titles from conference websites and summarizes them with chatGPT API calls.

## Instructions
After creating a python virtual environment.
1. ```pip install -r requirements.txt```
2. Create config.py file with line
```python
OPENAI_API_KEY = "ABCD"
```
3. ```python marl.py```

## Tips and Tricks
1. To change parser for a conference override `Conference.get_titles()` method.
2. To change prompt for a new topic inherit from Paper and override `prompt` property.
3. To change how output from chat agent is processed override `ChatAgent.process_result()` method.