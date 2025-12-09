import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv 
import urllib.parse
from urllib.parse import urlparse
import requests
from io import BytesIO
import socket
import openai
import torch
from typing import Optional
from PIL import Image
import dashscope
from dashscope import ImageSynthesis

# ==================== 1. 从环境变量加载设置 ====================
load_dotenv() 

# 从环境变量读取，如果不存在则使用空字符串（防止报错）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen-max")  # 使用兼容的默认模型

# 检查关键密钥是否已设置
if not LLM_API_KEY:
    raise ValueError("请在 .env 文件或环境变量中设置 'LLM_API_KEY'")

# 初始化LLM模型
llm = ChatOpenAI(
    model=LLM_MODEL_NAME,
    openai_api_key=LLM_API_KEY,
    openai_api_base=LLM_BASE_URL if LLM_BASE_URL else None,
    temperature=0.7,
)

dashscope.api_key = LLM_API_KEY

# 定义图像生成函数 (增强错误处理版本)
def generate_image_from_prompt(prompt: str) -> Optional[Image.Image]:
    """
    使用通义千问 Qwen-Image API 生成图像，并实现异步轮询。
    返回PIL Image对象，如果生成失败则返回None。
    """
    import time
    
    try:
        print(f"[Qwen-Image API] 提交任务，提示词: {prompt[:80]}...")

        # 1. 提交异步生成任务
        resp = ImageSynthesis.async_call(
            model='qwen-image-plus',  # 或 'qwen-image'
            prompt=prompt,
            size='1664*928',  # 16:9 横版，对应你的 `(Landscape poster)`
            n=1,
            prompt_extend=False
        )

        # 检查初始响应是否成功
        if resp.status_code != 200 or not hasattr(resp, 'output') or not hasattr(resp.output, 'task_id'):
            error_msg = getattr(resp, 'message', f'HTTP {resp.status_code}')
            print(f"[Qwen-Image API] 任务提交失败: {error_msg}")
            return None

        task_id = resp.output.task_id
        print(f"[Qwen-Image API] 任务提交成功，任务ID: {task_id}")

        # 2. 轮询任务状态，直到完成、失败或超时
        max_wait_time = 120  # 最大等待时间（秒），根据免费额度性能调整
        poll_interval = 3    # 轮询间隔（秒）
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            # 查询任务状态
            status_resp = ImageSynthesis.fetch(task_id)
            
            if status_resp.status_code != 200:
                print(f"[Qwen-Image API] 查询任务状态失败: {status_resp.status_code}")
                break

            task_status = status_resp.output.task_status
            print(f"[Qwen-Image API] 轮询中... 任务状态: {task_status}")

            if task_status == 'SUCCEEDED':
                # 任务成功，获取结果
                if hasattr(status_resp.output, 'results') and status_resp.output.results:
                    image_url = status_resp.output.results[0].url
                    print(f"[Qwen-Image API] 图像生成成功，开始下载...")
                    # 下载图片
                    image_response = requests.get(image_url, timeout=30)
                    if image_response.status_code == 200:
                        image = Image.open(BytesIO(image_response.content))
                        print("✅ 图像下载并转换成功")
                        return image
                    else:
                        print(f"[Qwen-Image API] 下载图片失败: {image_response.status_code}")
                        return None
                else:
                    print("[Qwen-Image API] 任务成功但无结果。")
                    return None
                    
            elif task_status == 'FAILED':
                # 任务失败
                error_msg = getattr(status_resp.output, 'message', '未知错误')
                print(f"[Qwen-Image API] 任务执行失败: {error_msg}")
                return None
                
            # 如果任务仍在运行或等待，则继续轮询
            elif task_status in ['PENDING', 'RUNNING']:
                time.sleep(poll_interval)
                continue
                
            else:
                # 遇到未知状态
                print(f"[Qwen-Image API] 任务进入未知状态: {task_status}")
                break

        # 循环结束，表示超时
        print(f"[Qwen-Image API] 错误：轮询超时（{max_wait_time}秒），任务可能仍在处理或已卡住。")
        return None

    except Exception as e:
        print(f"[Qwen-Image API] 图像生成过程发生未预期错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None

def network_test():
    """测试Space容器的网络连接"""
    results = []
    
    # 测试1：测试Vercel代理
    try:
        proxy_url = LLM_BASE_URL
        response = requests.post(
            proxy_url,
            json={"model": "qwen-max", "messages": [{"role": "user", "content": "test"}]},
            timeout=10
        )
        results.append(f"✅ 代理连接成功: 状态码 {response.status_code}")
    except Exception as e:
        results.append(f"❌ 代理连接失败: {str(e)}")
    
    # 测试2：测试DNS解析
    if LLM_BASE_URL:
        try:
            # 使用urlparse提取域名
            parsed_url = urlparse(LLM_BASE_URL)
            domain = parsed_url.netloc  # 这将得到类似 "qwen-proxy-psi.vercel.app"
            
            # 如果URL中包含端口号，需要去掉端口部分
            if ':' in domain:
                domain = domain.split(':')[0]
                
            ip = socket.gethostbyname(domain)
            results.append(f"✅ DNS解析成功: {domain} → {ip}")
        except Exception as e:
            results.append(f"❌ DNS解析失败 ({domain}): {e}")
    else:
        results.append("⚠️  LLM_BASE_URL未设置，跳过DNS解析测试")
    
    # 测试3：测试基础网络连通性
    try:
        response = requests.get("https://httpbin.org/ip", timeout=5)
        results.append(f"✅ 基础网络正常: {response.status_code}")
    except Exception as e:
        results.append(f"❌ 基础网络异常: {e}")
    
    # 测试4：测试是否被防火墙阻挡
    try:
        # 尝试访问不同端口
        response = requests.get("https://httpbin.org/headers", timeout=5)
        results.append(f"✅ HTTP请求正常: {response.status_code}")
    except Exception as e:
        results.append(f"❌ HTTP请求失败: {e}")
    
    return "\n".join(results)

# ==================== 2. 定义三个智能体的提示模板 ====================
# 智能体1：拆解智能体 - 理解需求，拆解核心元素
decompose_prompt = ChatPromptTemplate.from_template(
    """
    你是一个资深的创意策划。请将用户模糊的创意描述，拆解成图像生成所需的具体、可执行的核心元素列表。
    用户描述：{user_input}
    请按以下格式输出，直接给出答案：
    **主题**：[用一句话概括核心主题]
    **主要元素**：[列出3-5个关键视觉元素，用逗号分隔]
    **氛围**：[描述画面整体氛围，如"科幻、温暖、肃穆"]
    """
)

# 创建可运行链：prompt -> llm
chain_decompose = decompose_prompt | llm

# 智能体2：优化智能体 - 将元素转化为专业图像提示词
optimize_prompt = ChatPromptTemplate.from_template(
    """
    你是一名专业的AI绘画提示词工程师。请根据以下创意拆解，将其优化成一段详细、高质量的英文AI绘画提示词。
    请遵循以下规则：
    1. 提示词为英文，描述细致。
    2. 包含画面主体、细节、风格、色调、构图、画质等。
    3. 直接输出提示词，不要额外解释。

    创意拆解：
    {decomposed}
    优化后的专业提示词：
    """
)
# 创建可运行链：prompt -> llm
chain_optimize = optimize_prompt | llm

# 智能体3：升级为“海报设计师”智能体 - 直接生成海报风格的提示词
review_prompt = ChatPromptTemplate.from_template(
    """
    You are a professional graphic designer creating posters for campus events with AI.

    【Core Task】
    Generate ONE concise, effective English prompt for an AI image model to create a complete poster based on the event description.

    【Input Analysis & Smart Decisions】
    1.  **Size & Orientation (CRITICAL):** Analyze the following Chinese or English keywords in the description to decide the poster format. Integrate the chosen format like `(portrait poster)` or `(wide landscape poster)` into your final prompt.
        - Keywords for **Portrait**: `竖版`, `竖向`, `portrait`, `vertical` -> Choose **portrait (9:16)**
        - Keywords for **Square**: `方型`, `方形`, `正方形`, `square` -> Choose **square (1:1)**
        - Keywords for **Landscape**: `横版`, `横向`, `landscape`, `wide` -> Choose **landscape (16:9)**
        - **If no keywords are found, default to landscape (16:9).**

    2.  **Layout & Content (STRUCTURED):** Your prompt must describe a poster with these clear sections:
        - **MAIN HEADER:** A dominant, clear title area at the top. **If the description contains a title, use it. If not, invent a compelling, relevant title** (e.g., "Neural Nexus: AI Lecture Series" for an AI talk).
        - **INFORMATION BLOCK:** A dedicated area with event details (time, date, venue). **If details are provided, use them. If not, fabricate plausible, specific details** (e.g., "Date: Apr 15 | Time: 6:00 PM | Location: University Hall 203").
        - **CENTRAL VISUAL:** **One single, strong, symbolic icon/graphic** representing the event's core idea (e.g., interlocking gears for collaboration, a stylized brain for psychology). DO NOT describe a complex scene.
        - **CLEAR TYPOGRAPHY ZONES:** Visually separate the header, info block, and background. Use phrases like "clear typography," "distinct text areas," "bold header."

    3.  **Style & Atmosphere (CREATIVE):**
        - **Base Style:** "vector illustration", "flat design", "modern minimalist poster" – ensuring clarity for the AI model.
        - **Color & Mood:** Choose a color palette fitting the event's nature (cool blues/grays for academic, warm vibrant colors for festivals/arts).
        - **Random Artistic Flair (IMPORTANT):** **Randomly select and integrate ONE** of these styles to add uniqueness: `pop art`, `retro vintage`, `cyberpunk glow`, `watercolor splash`, `linocut print`.

    【Strict Output Rules】
    - Output **ONLY the final image generation prompt**. No explanations, prefixes, or additional text.
    - The prompt must be in **English**.
    - **Strictly limit to 70 English words.** Be concise and powerful.

    【Example Prompt Structure】
    "(Portrait poster) with a bold header 'AI Symposium 2024' and a lower info block stating 'Date: Nov 20 | Venue: Tech Center'. Central visual of a glowing, interconnected network nodes. Clean vector illustration, flat design with a cool blue and purple gradient, in a retro vintage style. Clear typography areas, minimalist layout."

    【Event Description to Analyze】
    {prompt}

    【Your Output (ONLY the image prompt)】
    """
)
# 创建可运行链：prompt -> llm
chain_review = review_prompt | llm

# ==================== 3. 重构：清晰、分步的协同工作流 ====================

def run_agent_chain(user_input: str):
    """
    分步执行智能体链，每一步都明确处理输入输出，易于调试。
    返回: (decomposed_text, optimized_prompt, final_prompt)
    """
    print(f"[STEP 0] 开始处理用户输入: {user_input}")
    
    # 第一步：拆解
    try:
        print(f"[STEP 1] 调用 chain_decompose...")
        # 明确构造输入字典
        step1_result = chain_decompose.invoke({"user_input": user_input})
        decomposed_text = step1_result.content
        print(f"[STEP 1] 成功。结果: {decomposed_text[:50]}...")  # 打印前50字符
    except Exception as e:
        print(f"[STEP 1] 失败: {e}")
        decomposed_text = f"拆解失败: {e}"
        return decomposed_text, "", ""  # 提前返回，因为后续步骤依赖此结果
    
    # 第二步：优化
    try:
        print(f"[STEP 2] 调用 chain_optimize...")
        # 明确使用上一步的结果作为输入
        step2_result = chain_optimize.invoke({"decomposed": decomposed_text})
        optimized_prompt = step2_result.content
        print(f"[STEP 2] 成功。结果: {optimized_prompt[:50]}...")
    except Exception as e:
        print(f"[STEP 2] 失败: {e}")
        optimized_prompt = f"优化失败: {e}"
        return decomposed_text, optimized_prompt, ""
    
    # 第三步：风格化 (升级为海报设计)
    try:
        print(f"[STEP 3] 调用海报设计师智能体...")
        # 直接将优化后的英文提示词传递给新的 review_prompt
        # 新的 prompt 将自行从中文关键词中解析尺寸、并补全信息
        step3_result = chain_review.invoke({"prompt": optimized_prompt}) # 注意变量名是 "prompt"
        final_prompt = step3_result.content.strip()
        
        word_count = len(final_prompt.split())
        print(f"[STEP 3] 海报提示词生成成功 (单词数: {word_count})。内容预览: {final_prompt[:80]}...")
        
    except Exception as e:
        print(f"[STEP 3] 失败: {e}")
        final_prompt = f"海报设计失败: {e}"
    
    return decomposed_text, optimized_prompt, final_prompt

# ==================== 4. 更新Gradio界面交互函数 ====================
def generate_poster(user_input):
    """ 处理用户输入，运行智能体链，并返回结果 """
    # 调用我们上面定义的分步函数
    decomposed_text, optimized_prompt, final_prompt_full = run_agent_chain(user_input)
    
    # 最终提示词
    final_image_prompt = final_prompt_full.strip()
    
    # 图像部分暂时为空
    generated_image = None
    if final_image_prompt and not final_image_prompt.startswith("风格化失败"):
        # 仅当成功获得提示词时才尝试生成图像
        print(f"[图像生成] 最终使用提示词 (长度{len(final_image_prompt.split())}词): {final_image_prompt[:60]}...")
        generated_image = generate_image_from_prompt(final_image_prompt)
    
    # 返回给Gradio显示
    # 注意：这里返回的是 decomposed_text, optimized_prompt, final_prompt_full
    return decomposed_text, optimized_prompt, final_prompt_full, generated_image

# ==================== 5. 构建并启动Gradio Web界面 ====================
with gr.Blocks(title="SynthPoster") as demo:
    gr.Markdown("# 🎨 智汇海报 海报创作智能体协同系统")
    gr.Markdown("体验三个AI智能体如何协同工作：拆解 → 优化 → 风格化")

    with gr.Row():
        with gr.Column():
            user_input = gr.Textbox(
                label="描述你想生成的海报",
                placeholder="例如：AI模型协同讲座",
                lines=3
            )
            btn = gr.Button("🚀 开始协同创作", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="生成的海报", width=512)

    with gr.Accordion("📝 点击查看智能体协同的完整过程", open=False):
        output_decomposed = gr.Textbox(label="智能体1 - 创意拆解", lines=3)
        output_optimized = gr.Textbox(label="智能体2 - 提示词优化", lines=3)
        output_final = gr.Textbox(label="智能体3 - 风格定稿", lines=3)

    # 绑定按钮点击事件
    btn.click(
        fn=generate_poster,
        inputs=[user_input],
        outputs=[output_decomposed, output_optimized, output_final, output_image]
    )
    
    # 添加测试部分
    gr.Markdown("## 网络诊断工具")
    test_btn = gr.Button("运行网络测试")
    test_output = gr.Textbox(label="测试结果", lines=10)
    test_btn.click(network_test, outputs=test_output)

     # ==================== 新增：API连通性测试功能区 ====================
    with gr.Accordion("🔧 API连通性测试（调试专用）", open=False):
        gr.Markdown("""
        **使用说明**：此功能将绕过LangChain，直接调用你配置的千问API。
        1. 点击测试按钮。
        2. 下方将显示：**你的配置**、**API原始响应**、**处理后的答案**。
        3. 如果失败，会显示具体错误，请核对配置（特别是Base URL和模型名）。
        """)
        test_btn = gr.Button("🧪 测试API连接", variant="secondary")
        test_output_config = gr.Textbox(label="你的API配置", lines=3, interactive=False)
        test_output_raw = gr.Textbox(label="API原始响应", lines=5, interactive=False)
        test_output_content = gr.Textbox(label="处理后的回答", lines=2, interactive=False)

        # 定义测试函数
        def test_api_connection():
            config_info = f"""正在测试的配置：
    API_KEY前5位: {LLM_API_KEY[:5] if LLM_API_KEY else 'None'}...
    BASE_URL: {LLM_BASE_URL}
    MODEL_NAME: {LLM_MODEL_NAME}
    """
            try:
                # 1. 初始化openai客户端
                client = openai.OpenAI(
                    api_key=LLM_API_KEY,
                    base_url=LLM_BASE_URL.rstrip('/')  # 移除末尾可能存在的斜杠
                )
                
                # 2. 发送一个简单的测试请求
                test_messages = [{"role": "user", "content": "请用中文简短回复：API连接测试成功。"}]
                # 设置明确的超时时间，避免长时间挂起
                response = client.chat.completions.create(
                    model=LLM_MODEL_NAME,
                    messages=test_messages,
                    temperature=0.7,
                    timeout=10.0  # 10秒超时
                )
                
                # 3. 整理并返回结果
                raw_response = f"响应对象类型: {type(response)}\n"
                raw_response += f"是否收到响应: {hasattr(response, 'choices')}\n"
                if hasattr(response, 'choices') and len(response.choices) > 0:
                    raw_response += f"choices 结构: {response.choices[0]}"
                
                answer = response.choices[0].message.content
                return config_info, raw_response, answer
                
            except openai.AuthenticationError as e:
                error_detail = f"{config_info}\n\n❌ 认证失败 (可能原因):\n1. API_KEY 无效或已过期\n2. 未开通对应模型服务\n3. 服务区域不正确\n\n错误详情: {e}"
                return error_detail, str(e), "认证失败"
            except openai.NotFoundError as e:
                error_detail = f"{config_info}\n\n❌ 未找到资源 (可能原因):\n1. MODEL_NAME '{LLM_MODEL_NAME}' 不正确\n2. BASE_URL 路径错误\n\n错误详情: {e}"
                return error_detail, str(e), "模型或端点不存在"
            except openai.APIConnectionError as e:
                error_detail = f"{config_info}\n\n🌐 网络连接失败 (可能原因):\n1. BASE_URL 无法访问\n2. 网络代理问题\n3. Hugging Face Space 容器网络限制\n\n错误详情: {e}"
                return error_detail, str(e), "网络连接失败"
            except Exception as e:
                error_detail = f"{config_info}\n\n⚠️ 未预期的错误:\n错误类型: {type(e).__name__}\n错误详情: {str(e)}"
                return error_detail, str(e), f"调用失败: {type(e).__name__}"
        
        # 绑定测试按钮事件
        test_btn.click(
            fn=test_api_connection,
            inputs=[],
            outputs=[test_output_config, test_output_raw, test_output_content]
        )

# 运行
if __name__ == "__main__":
    # 判断是否在 Hugging Face Space 环境中运行
    if os.getenv("SPACE_ID") is not None:
        # 🚀 Space 环境：使用默认的 launch() 配置，无需任何参数
        # Space 会自动处理网络、端口等所有配置
        demo.launch()
    else:
        # 💻 本地开发环境：使用你原来的配置
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)