import os
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live
from rich.text import Text
from provider import OpenAICompatibleClient
from tools.available_tools import available_tools
from prompt.system_prompt import AGENT_SYSTEM_PROMPT
from agent.parser import truncate_llm_output, parse_action, parse_finish, parse_tool_call

# --- 1. 加载配置 ---
load_dotenv('config/config.env')

API_KEY = os.getenv('API_KEY', '')
BASE_URL = os.getenv('BASE_URL', '')
MODEL_ID = os.getenv('MODEL_ID', '')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '')
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

console = Console()

llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 2. 用户交互循环 ---
while True:
    user_prompt = Prompt.ask("[bold cyan]请输入你的问题[/bold cyan]").strip()
    if not user_prompt or user_prompt.lower() in ("exit", "quit"):
        console.print("[bold green]再见！[/bold green]")
        break

    console.rule("[bold cyan]新对话[/bold cyan]", style="cyan")

    history: list[dict] = [
        {"role": "user", "content": user_prompt}
    ]

    console.print(Panel(
        f"[white]{user_prompt}[/white]",
        title="[bold]用户输入[/bold]",
        border_style="cyan"
    ))

    # --- 3. 运行主循环 ---
    for i in range(5):
        console.rule(f"[bold]第 {i+1} 轮[/bold]", style="dim")

        messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + history

        # 3.1 流式调用 LLM
        collected = ""
        with Live(
            Text("思考中...", style="cyan"),
            refresh_per_second=10,
            vertical_overflow="visible"
        ) as live:
            for token in llm.chat_stream(messages):
                collected += token
                live.update(Text(collected))

        # 3.2 截断多余的 Thought-Action 对
        collected = truncate_llm_output(collected)

        # 记录本轮 LLM 输出（assistant 角色）
        history.append({"role": "assistant", "content": collected})

        # 3.3 解析并执行行动
        action_str = parse_action(collected)
        if not action_str:
            console.print(Panel(
                "[red]错误: 未能解析到 Action 字段。[/red]",
                title="[red]解析错误[/red]",
                border_style="red"
            ))
            history.append({"role": "user", "content": "Observation: 错误: 未能解析到 Action 字段。"})
            continue

        if action_str.startswith("Finish"):
            final_answer = parse_finish(action_str)
            if final_answer:
                console.print(Panel(
                    f"[bold green]{final_answer}[/bold green]",
                    title="[bold green]✓ 最终答案[/bold green]",
                    border_style="green"
                ))
            break

        parsed = parse_tool_call(action_str)
        if not parsed:
            history.append({"role": "user", "content": "Observation: 错误: 无法解析工具调用格式。"})
            continue

        tool_name, kwargs = parsed

        if tool_name in available_tools:
            console.print(f"[yellow]⚡ 调用工具: {tool_name}{kwargs}[/yellow]")
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        console.print(Panel(
            f"[green]{observation}[/green]",
            title="[green]📡 观察结果[/green]",
            border_style="green"
        ))
        history.append({"role": "user", "content": f"Observation: {observation}"})
