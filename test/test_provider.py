"""测试 provider.OpenAICompatibleClient。"""

from provider import OpenAICompatibleClient


class TestOpenAICompatibleClient:
    def test_init(self):
        """客户端初始化正确保存参数。"""
        client = OpenAICompatibleClient(
            model="gpt-4",
            api_key="sk-test",
            base_url="https://api.test.com/v1",
        )
        assert client.model == "gpt-4"
        assert client.client.api_key == "sk-test"
        assert "api.test.com" in str(client.client.base_url)

    def test_generate_stream_returns_generator(self):
        """chat_stream 是生成器函数。"""
        client = OpenAICompatibleClient(
            model="test", api_key="test", base_url="https://test.com/v1"
        )
        messages = [{"role": "system", "content": "你是个助手。"},
                    {"role": "user", "content": "你好"}]
        gen = client.chat_stream(messages)
        # 验证它是一个生成器
        assert hasattr(gen, "__next__")
        assert hasattr(gen, "__iter__")
