import re
from typing import Optional


def truncate_llm_output(text: str) -> str:
    """截断 LLM 输出中多余的 Thought-Action 对，只保留第一对。"""
    match = re.search(
        r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)',
        text, re.DOTALL
    )
    if match:
        truncated = match.group(1).strip()
        if truncated != text.strip():
            return truncated
    return text


def parse_action(text: str) -> Optional[str]:
    """从 LLM 输出中解析 Action 行内容，未找到时返回 None。"""
    match = re.search(r"Action: (.*)", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_finish(text: str) -> Optional[str]:
    """从 Action 内容中解析 Finish[...] 的最终答案，未找到时返回 None。"""
    match = re.search(r"Finish\[(.*)\]", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_tool_call(action_str: str) -> Optional[tuple[str, dict[str, str]]]:
    """从 Action 内容中解析工具名和参数字典，格式不合法时返回 None。

    预期格式: tool_name(arg1="value1", arg2="value2")
    """
    tool_match = re.search(r"(\w+)\(", action_str)
    args_match = re.search(r"\((.*)\)", action_str)
    if not tool_match or not args_match:
        return None
    tool_name = tool_match.group(1)
    args_str = args_match.group(1)
    kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
    return tool_name, kwargs
