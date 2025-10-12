"""测试 tools 模块的工具函数（模拟 HTTP 调用）。"""

import requests
from tools.get_weather import get_weather
from tools.get_attraction import get_attraction


class TestGetWeather:
    def test_success(self, mocker):
        """天气查询成功时返回格式化文本。"""
        fake_response = {
            "current_condition": [
                {"weatherDesc": [{"value": "Sunny"}], "temp_C": "28"}
            ]
        }
        mock_get = mocker.patch("requests.get")
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = fake_response

        result = get_weather("重庆")
        assert "重庆" in result
        assert "Sunny" in result
        assert "28" in result
        mock_get.assert_called_once_with("https://wttr.in/重庆?format=j1")

    def test_network_error(self, mocker):
        """网络异常（如超时）时返回友好错误信息。"""
        mocker.patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("连接超时"),
        )
        result = get_weather("重庆")
        assert "错误" in result
        assert "连接超时" in result

    def test_invalid_city(self, mocker):
        """API 返回非 200 状态码时走错误路径。"""
        mock_get = mocker.patch("requests.get")
        mock_get.return_value.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("City not found")
        )

        result = get_weather("InvalidCity")
        assert "错误" in result

    def test_malformed_response(self, mocker):
        """API 返回的 JSON 缺少关键字段时返回友好错误。"""
        mock_get = mocker.patch("requests.get")
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"no_data": True}

        result = get_weather("重庆")
        assert "错误" in result


class TestGetAttraction:
    def _patch_tavily(self, mocker, return_value=None, side_effect=None):
        """辅助方法：mock TavilyClient 及其 search 方法。"""
        mock_client = mocker.patch("tools.get_attraction.TavilyClient")
        mock_instance = mock_client.return_value
        if side_effect:
            mock_instance.search.side_effect = side_effect
        else:
            mock_instance.search.return_value = return_value
        return mock_instance

    def test_with_answer(self, mocker):
        """Tavily 返回 answer 时直接使用。"""
        fake_response = {
            "answer": "推荐洪崖洞夜景、解放碑步行街。",
            "results": [
                {"title": "洪崖洞", "content": "夜景著名景点"}
            ],
        }
        self._patch_tavily(mocker, return_value=fake_response)
        mocker.patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})

        result = get_attraction("重庆", "Sunny")
        assert "洪崖洞" in result
        assert "解放碑" in result

    def test_no_answer_with_results(self, mocker):
        """Tavily 没有 answer 但有 results 时格式化展示。"""
        fake_response = {
            "answer": None,
            "results": [
                {"title": "南山公园", "content": "俯瞰城市全景"}
            ],
        }
        self._patch_tavily(mocker, return_value=fake_response)
        mocker.patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})

        result = get_attraction("重庆", "Sunny")
        assert "南山公园" in result

    def test_no_results(self, mocker):
        """没有搜索结果时返回提示信息。"""
        fake_response = {"answer": None, "results": []}
        self._patch_tavily(mocker, return_value=fake_response)
        mocker.patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})

        result = get_attraction("重庆", "Sunny")
        assert "没有找到" in result

    def test_api_error(self, mocker):
        """Tavily 调用异常时返回友好错误。"""
        self._patch_tavily(mocker, side_effect=Exception("API 调用失败"))
        mocker.patch.dict("os.environ", {"TAVILY_API_KEY": "test-key"})

        result = get_attraction("重庆", "Sunny")
        assert "错误" in result

    def test_missing_api_key(self, mocker):
        """缺少 API key 时返回提示。"""
        mocker.patch.dict("os.environ", {}, clear=True)

        result = get_attraction("重庆", "Sunny")
        assert "未配置" in result
