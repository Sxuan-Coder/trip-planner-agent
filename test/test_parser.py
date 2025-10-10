"""测试 agent.parser 模块中的 ReAct 解析函数。"""

from agent.parser import (
    truncate_llm_output,
    parse_action,
    parse_finish,
    parse_tool_call,
)


class TestTruncateLlmOutput:
    def test_single_pair_unchanged(self):
        """单对 Thought-Action 保持不变。"""
        text = "Thought: 需要查询天气。\nAction: get_weather(city=\"北京\")"
        assert truncate_llm_output(text) == text

    def test_truncate_trailing_pair(self):
        """多余的第二对 Thought-Action 被截断。"""
        text = (
            "Thought: 需要查询天气。\nAction: get_weather(city=\"北京\")\n"
            "Thought: 还需要做什么？\nAction: get_weather(city=\"上海\")"
        )
        result = truncate_llm_output(text)
        assert result == "Thought: 需要查询天气。\nAction: get_weather(city=\"北京\")"

    def test_truncate_with_trailing_observation(self):
        """如果第一对后面紧跟 Observation 则保留下面的内容也被保留(实际不会出现Observation在前面的场景),
        但测试确保截断在Action处停止。
        """
        text = "Thought: 查天气。\nAction: get_weather(city=\"北京\")\nObservation: 晴天"
        result = truncate_llm_output(text)
        assert "Observation" not in result

    def test_truncate_clean_newlines(self):
        """截断结果 strip 前后空白。"""
        text = "Thought: 查天气。\nAction: get_weather(city=\"北京\")\n   \nThought: 多余的。\nAction: 结束"
        result = truncate_llm_output(text)
        assert result == "Thought: 查天气。\nAction: get_weather(city=\"北京\")"

    def test_no_action_unchanged(self):
        """不包含 Action 的文本原样返回。"""
        text = "Thought: 一些随机的思考内容"
        assert truncate_llm_output(text) == text.strip()

    def test_empty_string(self):
        """空字符串不报错。"""
        assert truncate_llm_output("") == ""


class TestParseAction:
    def test_basic_action(self):
        """正常解析 Action 行。"""
        text = "Thought: 查天气。\nAction: get_weather(city=\"北京\")"
        assert parse_action(text) == 'get_weather(city="北京")'

    def test_action_with_newlines(self):
        """Action 内容中不含换行（通常在同一行）。"""
        text = "Thought: 思考。\nAction: get_weather(city=\"重庆\")"
        assert parse_action(text) == 'get_weather(city="重庆")'

    def test_no_action_returns_none(self):
        """没有 Action 时返回 None。"""
        text = "Thought: 一些思考内容"
        assert parse_action(text) is None

    def test_empty_string(self):
        """空字符串返回 None。"""
        assert parse_action("") is None


class TestParseFinish:
    def test_basic_finish(self):
        """解析 Finish[] 中的内容。"""
        assert parse_finish('Finish[晴天，25°C]') == '晴天，25°C'

    def test_finish_with_multiline(self):
        """Finish[] 内容包含换行。"""
        text = """Finish[推荐行程：
- 上午：解放碑
- 下午：南山公园
建议做好防晒。]"""
        result = parse_finish(text)
        assert "推荐行程：" in result
        assert "上午：解放碑" in result
        assert "建议做好防晒。" in result

    def test_no_finish_returns_none(self):
        """不含 Finish 时返回 None。"""
        assert parse_finish("get_weather(city=\"重庆\")") is None

    def test_finish_with_nested_brackets(self):
        """内容中包含其余括号不影响解析。"""
        result = parse_finish('Finish[天气(晴朗)，温度25°C]')
        assert result == '天气(晴朗)，温度25°C'


class TestParseToolCall:
    def test_basic_tool_call(self):
        """解析工具名和参数。"""
        name, kwargs = parse_tool_call('get_weather(city="北京")')
        assert name == "get_weather"
        assert kwargs == {"city": "北京"}

    def test_multiple_args(self):
        """多个参数被正确解析。"""
        name, kwargs = parse_tool_call('get_attraction(city="重庆", weather="Sunny")')
        assert name == "get_attraction"
        assert kwargs == {"city": "重庆", "weather": "Sunny"}

    def test_no_match_returns_none(self):
        """格式不匹配时返回 None。"""
        assert parse_tool_call("just some text") is None

    def test_empty_string(self):
        """空字符串返回 None。"""
        assert parse_tool_call("") is None
