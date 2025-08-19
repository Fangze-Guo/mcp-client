import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Optional

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

# 加载 .env 文件
load_dotenv()


class MCPClient:
    def __init__(self):
        """初始化 MCP 客户端"""
        self.messages = [
            {"role": "system",
             "content": """
             你需要根据用户的问题调用天气工具，获取结果后，必须用自然语言整理温度、湿度、天气状况等信息回复用户。
             - 如果工具返回了数据，禁止生成无关内容（如询问其他需求）。
             - 如果用户的问题与天气无关，直接用自然语言回答，不要调用工具。
             """
             }
        ]
        self.exit_stack = AsyncExitStack()
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("未找到 OpenAI API Key")

        # 创建 OpenIA client
        self.client = OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None

    async def connect_to_server(self, server_script_path: str):
        """连接到 MCP 服务器并列出可用工具"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("服务器脚本必须是 .py 或 .js 文件")

        # 确定启动命令（Python 用 "python"，JS 用 "node"）
        command = "python" if is_python else "node"
        # 构建服务器启动参数
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        # 启动 MCP 服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        # 初始化客户端会话（与服务器建立正式连接）
        await self.session.initialize()

        # 列出 MCP 服务器上的工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n已连接到服务器，支持以下工具：", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """
        使用大模型处理查询并调用可用的 MCP 工具（Function Calling）
        :param query: 用户查询字符串
        :return: 最终回答字符串
        """
        # 将用户查询添加到历史上下文
        self.messages.append({"role": "user", "content": query})

        # 从 MCP 服务器获取所有可用工具
        response = await self.session.list_tools()

        # 将工具格式化为大模型可理解的 Function Calling 格式
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        # 调用模型时使用累积的对话历史
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=available_tools
        )

        # 处理返回内容
        content = response.choices[0]
        if content.finish_reason == "tool_calls":
            # 如果是需要使用工具，就解析工具
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            # 执行工具
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")

            # 将工具调用指令添加到历史上下文
            self.messages.append(content.message.model_dump())
            # 将工具返回结果添加到历史上下文
            self.messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id
            })

            # 将上面的结果返回给大模型用于生成最终结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages
            )
            answer = response.choices[0].message.content
            # 将模型回答添加到历史
            self.messages.append({"role": "assistant", "content": answer})
            return answer
        # 如果模型无需调用工具（finish_reason不是tool_calls），直接返回模型的回答
        else:
            answer = content.message.content
            self.messages.append({"role": "assistant", "content": answer})
            return answer

    async def chat_loop(self):
        """运行交互式聊天"""
        print("\nMCP 客户端已启动，输入 'quit' 退出")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print(f"\nOpenAI: {response}")
            except Exception as e:
                print(f"发生错误：{str(e)}")

    async def cleanup(self):
        """清理资源"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await  client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
