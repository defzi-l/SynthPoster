import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv 
import urllib.parse
from urllib.parse import urlparse
import requests
import socket
import openai
import torch
from diffusers import AutoPipelineForText2Image
from typing import Optional
from PIL import Image

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

# 选择适合CPU的模型，并在内存不足时进行优化
IMAGE_MODEL_ID = "stabilityai/sd-turbo"  # 比SDXL-Turbo更轻量的模型

# 动态加载图像生成管道，根据容器性能选择配置
try:
    # 尝试加载fp16精度的模型以节省内存，如果失败则降级为fp32
    try:
        image_pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,  # 禁用安全检查器以加速并减少内存占用
            use_safetensors=True
        )
    except (RuntimeError, OSError):
        # 如果fp16失败，可能是内存不足或设备不支持，回退到fp32
        print("fp16加载失败，尝试使用fp32精度加载模型...")
        image_pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL_ID,
            torch_dtype=torch.float32,
            safety_checker=None,
            use_safetensors=True
        )
    
    # 将模型移至CPU（Hugging Face Space免费容器为CPU环境）
    image_pipe = image_pipe.to("cpu")
    
    # 启用CPU优化，大幅减少内存使用并加速推理[citation:7][citation:8]
    image_pipe.enable_attention_slicing()  # 注意力切片，降低峰值内存
    if hasattr(image_pipe, "enable_cpu_offload"):
        image_pipe.enable_cpu_offload()  # 如果管道支持CPU卸载，则启用
    
    print("✅ 图像生成模型加载成功 (运行在CPU模式)")
    
except Exception as e:
    print(f"❌ 图像生成模型加载失败: {e}")
    image_pipe = None

# 定义图像生成函数
def generate_image_from_prompt(prompt: str) -> Optional[Image.Image]:
    """
    使用加载的模型根据提示词生成图像。
    返回PIL Image对象，如果生成失败则返回None。
    """
    if image_pipe is None:
        print("图像生成模型未加载，无法生成图片。")
        return None
    
    try:
        print(f"[Qwen-Image API] 提交任务，提示词: {prompt[:80]}...")

        # 根据提示词中的关键词动态设置尺寸
        # Qwen-Image 支持的标准尺寸映射
        size_map = {
            'portrait': '928*1664',   # 9:16 竖版 (默认)
            'square': '1328*1328',    # 1:1 方形
            'landscape': '1664*928'   # 16:9 横版
        }
        # 检测提示词中的版式关键词
        prompt_lower = prompt.lower()
        chosen_size = size_map['portrait']  # 默认竖版
        for key in size_map:
            if key in prompt_lower:
                chosen_size = size_map[key]
                print(f"[尺寸映射] 检测到 '{key}'，使用尺寸: {chosen_size}")
                break

        # 1. 提交异步生成任务
        resp = ImageSynthesis.async_call(
            model='qwen-image-plus',  # 或 'qwen-image'
            prompt=prompt,
            size=chosen_size, 
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
        print(f"❌ 图像生成过程出错: {e}")
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

# ==================== 重构：智能体3 - 千问海报设计师智能体 ====================
# 此智能体直接分析用户原始描述，生成专为Qwen-Image优化的中英混合海报提示词。
review_prompt = ChatPromptTemplate.from_template(
    """
你是一名专业海报设计师，专门为通义千问AI文生图模型（Qwen-Image）设计生成提示词。

【你的核心任务】
根据用户对活动的原始中文描述，生成一段详细的、中英混合的AI图像生成提示词，以创建一张信息完整、视觉突出的**中文校园活动海报**。

【关键信息提取与结构化 (从用户输入中)】
请严格按以下步骤分析用户输入：
1.  **提取或生成标题**：如果描述中提供了活动标题（如“《模型协同》学术讲座”），直接提取。否则，基于活动主题生成一个简洁、有力的**中文主标题**（例如“AI融合创新论坛”）。
2.  **提取或补全信息**：
    - **时间**：必须提取或推断出具体的日期、开始和结束时间（如“2025年12月20日 下午3:00-5:00”）。如果只有“下午3点”，请补全为“下午3:00开始”。
    - **地点**：提取具体地点（如“科学会堂101”）。如果未提供，则根据活动类型生成一个合理的**中文地点**（如“大学生活动中心”）。
3.  **决定海报版式**：分析描述中的关键词，决定海报形状，并在你的提示词开头用英文注明：
    - 如果包含 `竖版`、`竖向`、`portrait`、`vertical` -> 使用 `(Portrait poster, 9:16 ratio)`
    - 如果包含 `方型`、`方形`、`square` -> 使用 `(Square poster, 1:1 ratio)`
    - 如果包含 `横版`、`横向`、`landscape`、`wide` -> 使用 `(Landscape poster, 16:9 ratio)`
    - **如果无关键词，默认使用 `(Portrait poster, 9:16 ratio)`**。

【构建你的提示词 (中英混合，结构清晰)】
按照以下结构和语言规则构建最终提示词：
1.  **海报版式与布局（英文）**：以第3步决定的版式英文描述开头，并描述布局：“Clear layout with distinct zones for title, information, and central visual.”
    " The poster design fills the entire frame with no borders or margins, edge-to-edge composition."
2.  **核心中文信息 - 文字精确性强化**：
    - **标题区域**：描述“A large, bold header at the top featuring the Chinese text: 【这里放入第1步得到的中文标题】”。
    - **信息区域**：描述“A clean information block below with the Chinese details: 【时间: 第2步得到的具体时间】|【地点: 第2步得到的具体地点】”。
    *注意：必须用【】标注出要生成的确切中文文字。*
    **文字生成规则**：
    - **字形要求**：`Ensure every Chinese character is written correctly, with no missing or extra strokes, no typos, and clear legibility.`
    - **字体风格**：`Use a clean, modern, and bold sans-serif font that is highly readable, similar to "Microsoft YaHei" or "PingFang SC". Avoid cursive or overly decorative fonts.`
    - **布局强化**：`The text should be centered, with high contrast against the background (e.g., white text on dark background or black text on light background).`
3.  **中央视觉与风格（英文）**：
    - **核心图形**：基于活动主题，描述一个**象征性的、简单的图形**，如“Central visual of a stylized, interconnected network of nodes (representing model collaboration)”。
    - **整体风格**：使用“Modern minimalist poster, flat vector illustration”。
    - **色调**：根据活动类型选择，如学术类用“cool blue and gray color palette”。
    - **随机艺术风格**：从以下列表中随机选择1-2种结合：`cyberpunk glow`, `retro vintage poster style`, `pop art, Roy Lichtenstein style`, `watercolor texture`, `pencil sketch`, `3D render, Blender`, `stained glass art`, `Chinese ink painting`, `low poly graphic`, `surrealism, Dali style`。
4.  **质量与清晰度（英文）**：以“High contrast, clear typography, suitable for print. --ar 16:9 --q 2”结尾。（`--ar` 后的比例根据版式调整）

【最终输出规则】
- **只输出**最终生成图像的完整提示词，**不要有任何额外解释**。
- 提示词总长度控制在**100-120个英文单词**以内。
- **严格遵循上述结构和语言混合要求**。

【用户原始描述】
{user_input}

【你的输出 (仅提示词)】
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
    
    # 第三步：风格化
    try:
        print(f"[STEP 3] 调用海报设计师智能体...")
        # 直接将优化后的英文提示词传递给新的 review_prompt
        # 新的 prompt 将自行从中文关键词中解析尺寸、并补全信息
        step3_result = chain_review.invoke({"user_input": user_input})
        final_prompt = step3_result.content.strip()
        
        word_count = len(final_prompt.split())
        print(f"[STEP 3] 海报提示词生成成功 (单词数: {word_count})。内容预览: {final_prompt[:80]}...")
        
    except Exception as e:
        print(f"[STEP 3] 失败: {e}")
        final_prompt = f"风格化失败: {e}"
    
    return decomposed_text, optimized_prompt, final_prompt

# ==================== 4. 更新Gradio界面交互函数 ====================
def generate_poster(user_input):
    """ 处理用户输入，运行智能体链，并返回结果 """
    # 调用我们上面定义的分步函数
    decomposed_text, optimized_prompt, final_prompt_full = run_agent_chain(user_input)
    
    # **在图像生成前，进行严格的失败检测**
    error_keywords = ["失败", "missing variables", "Error", "Exception", "Traceback"]
    # 检查最终提示词是否包含任何错误关键词
    if any(keyword in final_prompt_full for keyword in error_keywords):
        # 如果检测到错误，立即停止，并返回错误信息，第四个返回值为None（无图片）
        error_msg = f"流程错误，已终止图像生成以避免浪费Token。错误信息：{final_prompt_full[:150]}..."
        print(f"[流程拦截] {error_msg}")
        return decomposed_text, optimized_prompt, error_msg, None

    # 最终提示词
    final_image_prompt = final_prompt_full.strip()
    
    # 图像部分暂时为空
    generated_image = None
    if final_image_prompt and not final_image_prompt.startswith("风格化失败"):
        # 仅当成功获得提示词时才尝试生成图像
        generated_image = generate_image_from_prompt(final_image_prompt)
    
    # 返回给Gradio显示
    # 注意：这里返回的是 decomposed_text, optimized_prompt, final_prompt_full
    return decomposed_text, optimized_prompt, final_prompt_full, image_output

# ==================== 5. 构建并启动Gradio Web界面 ====================
with gr.Blocks(title="SynthPoster", css=".scrollable-textbox textarea {overflow-y: auto !important;}") as demo:
    gr.Markdown("# 🎨 智汇海报 海报创作智能体协同系统")
    gr.Markdown("体验三个AI智能体如何协同工作：拆解 → 优化 → 风格化")

    with gr.Row():
        with gr.Column(scale=2):
            user_input = gr.Textbox(
                label="描述你想生成的海报",
                placeholder="例如：AI模型协同讲座",
                lines=3
            )
            btn = gr.Button("🚀 开始协同创作", variant="primary")

        with gr.Column(scale=1):
            # 固定图片尺寸为竖版海报比例
            output_image = gr.Image(
                label="生成的海报",
                width=360,        # 竖版宽度稍小
                height=512,       # 竖版高度
                scale=0           # 确保图片缩放适应区域
            )

    with gr.Accordion("📝 点击查看智能体协同的完整过程", open=False):
        # 为三个文本框添加滚动条
        output_decomposed = gr.Textbox(
            label="智能体1 - 创意拆解",
            lines=3,
            interactive=False,
            elem_classes=["scrollable-textbox"]
        )
        output_optimized = gr.Textbox(
            label="智能体2 - 提示词优化",
            lines=3,
            interactive=False,
            elem_classes=["scrollable-textbox"]
        )
        output_final = gr.Textbox(
            label="智能体3 - 风格定稿 (最终提示词)",
            lines=3,
            interactive=False,
            elem_classes=["scrollable-textbox"]
        )

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
                    temperature=0.8,
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