import json
import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("WeatherServer")

# 加载 .env 文件
load_dotenv()

OPENWEATHER_API_BASE = os.getenv("OPENWEATHER_API_BASE")
API_KEY = os.getenv("API_KEY")
USER_AGENT = os.getenv("USER_AGENT")


async def fetch_weather(city: str) -> dict[str, Any] | None:
    """
    从 OpenWeather API 获取天气信息。
    :param city: 城市名称
    :return: 天气数据字典
    """
    # API 请求参数配置
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "zh_cn"
    }
    headers = {"User-Agent": USER_AGENT}

    async with httpx.AsyncClient() as client:
        try:
            # 发送异步GET请求，等待响应
            response = await client.get(
                OPENWEATHER_API_BASE,
                params=params,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"error": f"HTTP 错误：{e.response.status_code}"}
        except Exception as e:
            return {"error": f"请求失败：{str(e)}"}


def format_weather(data: dict[str, Any] | str) -> str:
    """
    将天气数据格式化为易读文本
    :param data: 天气数据
    :return: 格式化后的天气信息
    """
    # 如果传入的是字符串，先转换为字典
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception as e:
            return f"无法解析天气数据：{e}"

    # 包含错误信息，直接返回提示
    if "error" in data:
        return f"{data['error']}"

    city = data.get("name", "未知")
    country = data.get("sys", {}).get("country", "未知")
    temp = data.get("main", {}).get("temp", "N/A")
    humidity = data.get("main", {}).get("humidity", "N/A")
    wind_speed = data.get("wind", {}).get("speed", "N/A")
    weather_list = data.get("weather", [{}])
    description = weather_list[0].get("description", "未知")

    return (
        f"🌍 {city}, {country}\n"
        f"🌡️ 温度：{temp}°C\n"
        f"💧 湿度：{humidity}%\n"
        f"🍃 风速：{wind_speed} m/s\n"
        f"🌤️ 天气：{description}\n"
    )


@mcp.tool()
async def query_weather(city: str) -> str:
    """
    输入指定城市的英文名称，返回当天天气查询结果
    :param city: 城市名
    :return: 格式化后的天气信息
    """
    if not city.strip():  # 检查城市名是否为空或仅空格
        return "错误：城市名称不能为空，请提供有效的城市名（英文）。"
    data = await fetch_weather(city)
    return format_weather(data)


if __name__ == "__main__":
    mcp.run(transport='stdio')
