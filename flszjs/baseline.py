# encoding=utf-8
import requests
import json
import os
import dashscope
from dashscope import Generation
from http import HTTPStatus
import time
import re
from tqdm import tqdm

def ask_llm(prompt, model="qwen3-235b-a22b-instruct-2507"):
    if ("qwen3-235b-a22b-instruct-2507" == model):
        return ask_tyqw_general(prompt,model)
    
def ask_tyqw_general(prompt, model='qwen3-235b-a22b-instruct-2507'):
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "fill_your_api_key")
    if type(prompt) is str:
        s_time = time.time()
        response = Generation.call(model,
                                   prompt=prompt,
                                   )
        e_time = time.time()
        
        # print(response)
        if response.status_code == HTTPStatus.OK:
            used_time = e_time-s_time
            input_tokens = response['usage'].input_tokens
            output_tokens = response['usage'].output_tokens
            token_infomration = {"used_time":used_time,"input_tokens":input_tokens,"output_tokens":output_tokens}
            return response["output"]["text"],token_infomration
        return None,{"used_time":None,"input_tokens":None,"output_tokens":None}

    elif type(prompt) is list:
        response = Generation.call(model,
                                   messages=prompt,
                                   result_format='message'  # 设置输出为'message'格式
                                   )
        if response.status_code == HTTPStatus.OK:
            return response["output"]["choices"][0]["message"]["content"]
        else:
            return None
        
# 使用正则表达式提取内容
def extract_python(text):
    if "```python" in text:
        pattern = r'```python\n(.*?)\n```'

        match = re.search(pattern, text, re.DOTALL)

        if match:
            extracted_content = match.group(1)

            return True,eval(extracted_content)
        else:
            return False,None
    else:
        try:
            return True,eval(text)
        except:
            return False,None
        
prompt = """针对查询中涉及到的问题回答数值计算结果并给出依据的法律条款：
{query}

要求：
1. 一步一步思考，最后给出答案
2. 最后返回一个python字典
```python
{"article_answer":[xx,..],
"numerical_answer":[1]}
```
其中，article_answer《xxx》第xx条。numerical_answer是一个列表，里面包括了具体的数值
"""

def process_res(num,model_name='qwen3-235b-a22b-instruct-2507'):
    query = data_test[num]['query']
    response, usage = ask_llm(prompt.replace("{query}",query)
                              ,model=model_name)
        
    return response


with open(f"./test.jsonl") as f:
    data_test = f.readlines()
data_test = [json.loads(i) for i in data_test]


pred_res = []

for i in tqdm(range(len(data_test))):
    res = process_res(i)
    pred_res.append(res)

final_res = []
for i in range(len(data_test)):
    _,item = extract_python(pred_res[i])
    item["reasoning_content"] = pred_res[i]
    item['id'] = i
    final_res.append(item)
    
with open(f"prediction.jsonl", 'w', encoding='utf-8') as f:
    # 使用列表推导式一次性写入所有行
    f.write('\n'.join(json.dumps(item, ensure_ascii=False) for item in final_res))