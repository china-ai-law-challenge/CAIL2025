import json
import os
from dashscope import Generation

prompt = """
#任务：
你是一个司法考试助手，请对司法考试的主观问题进行回答。

#要求：
1.回复务必使用法言法语，保证语言简洁、逻辑清晰。
2.如果有多个问题，则直接回答每个问题，不得重复问题。
3.要求输出内容无格式，且输出长度不超过200个字。

#输入：
{question}

#输出
现在请开始你的回答。
"""

def ask_llm(model, messages):
    """
    call your qwen_api and return response
    example: call qwen api from Bailian (https://help.aliyun.com/zh/model-studio/getting-started/what-is-model-studio?spm=a2c4g.11174283.0.i3)
    """
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY", "sk-3094f578e6284ae4a7cf45eb722187c8"),
        model=model,
        messages=messages,
        enable_thinking=False,
        result_format='message'
    )
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None


def predict(model, input_file, output_file):
    """
    :param input_file:
    :param output_file:
    """
    lines = open(input_file).readlines()
    with open(output_file, 'w') as f_out:
        for line in lines:
            data = json.loads(line)
            question = data['big_ques'] + data['small_ques']
            messages = [{"role": "user", "content": prompt.format(question=question)}]
            user_answer = ask_llm(model, messages)
            j = {"id": data['id'], "user_answer": user_answer}
            print(j)
            f_out.write(json.dumps(j, ensure_ascii=False) + "\n")
            f_out.flush()


if __name__ == '__main__':
    model = "qwen3-32b"
    predict(model, './test_data.json', './prediction.json')