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

# 智能体3：审查/风格化智能体 - 为提示词添加统一风格
review_prompt = ChatPromptTemplate.from_template(
    """
    你是一名专业的校园活动艺术总监。请根据用户对活动海报的描述，判断其活动类型，并为其优化和定型AI绘画提示词，使其符合该类校园海报的专业风格。

    请严格按以下步骤执行：
    1. **判断活动类型**：根据描述，从以下常见类型中选择最匹配的，或推断一个合理的类型：
       - **学术类**（如讲座、研讨会、竞赛）
       - **招募类**（如社团招新、志愿者招募、队员招募）
       - **文艺类**（如音乐会、话剧、舞蹈演出、画展）
       - **庆典节日类**（如迎新晚会、毕业季、圣诞派对、校庆）
       - **体育健身类**（如运动会、篮球赛、马拉松、瑜伽课）
       - **宣传倡导类**（如环保倡议、公益宣传、心理健康周）

    2. **优化与定型**：
       - 保持用户描述的**核心元素和原意**。
       - 将语言优化得更富有**视觉冲击力、感染力和青春气息**，适合海报传播。
       - 根据你判断的活动类型，在提示词末尾**自动添加最匹配的风格后缀**。

    3. **添加风格后缀示例**
       - 学术类：`, academic poster, clean layout, infographic style, vector illustration, vibrant, 4k`
       - 招募类：`, recruitment poster, dynamic composition, bold typography, team spirit, flat design, vibrant colors`
       - 文艺类：`, artistic poster, dramatic lighting, creative, painting style, trending on artstation, 8k`
       - 庆典类：`, festive poster, joyful atmosphere, confetti, glowing lights, vector art, bright color palette`
       - 体育类：`, sports poster, action shot, motion blur, energetic, strong contrast, graphic design`
       - 宣传倡导类：`, public awareness poster, symbolic, minimalist, powerful message, solid background`

    【用户描述】
    {prompt}

    【你的输出】
    请直接输出以下两部分内容，用"---"分隔：
    第一部分：仅一句话说明"判断为：【类型】类活动海报"。
    第二部分：直接给出优化并添加了对应风格后缀的完整英文提示词。
    """
)
# 创建可运行链：prompt -> llm
chain_review = review_prompt | llm

# ==================== 3. 将智能体串联成协同工作流 ====================
# 定义完整的处理流水线
overall_chain = RunnableSequence(
    # 第一步：接收初始输入，传递给拆解链
    RunnablePassthrough.assign(decomposed_text=lambda x: chain_decompose.invoke(x)),
    # 第二步：使用上一步的输出去优化
    lambda x: {"optimized_prompt": chain_optimize.invoke(x["decomposed_text"])},
    # 第三步：使用优化后的输出去审查/风格化
    lambda x: {"final_prompt": chain_review.invoke(x["optimized_prompt"])}
)

# ==================== 4. 定义Gradio界面交互函数 ====================
def generate_poster(user_input):
    """ 处理用户输入，运行智能体链，并返回结果 """
    try:
        # 1. 运行整个协同链
        # 现在 overall_chain 是一个 RunnableSequence，可直接调用
        result = overall_chain.invoke({"user_input": user_input})

        # 2. 从结果字典中安全地提取每个环节的文本内容
        # 注意：chain.invoke() 返回的是 AIMessage 对象，需要用 .content 获取文本
        decomposed_text = result.get("decomposed_text", "").content if hasattr(result.get("decomposed_text"), 'content') else str(result.get("decomposed_text", ""))
        optimized_prompt = result.get("optimized_prompt", "").content if hasattr(result.get("optimized_prompt"), 'content') else str(result.get("optimized_prompt", ""))
        final_prompt = result.get("final_prompt", "").content if hasattr(result.get("final_prompt"), 'content') else str(result.get("final_prompt", ""))

        # 3. （后续）此处应调用图像生成API，用 final_prompt 生成图片
        image_output = None

        # 4. 返回给Gradio显示
        return decomposed_text, optimized_prompt, final_prompt, image_output

    except Exception as e:
        # 异常处理：将错误信息返回给界面，方便调试
        error_msg = f"处理过程中出现错误：{str(e)}"
        return error_msg, error_msg, error_msg, None

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

    gr.Markdown("### 💡 说明：当前使用占位图片。集成图像API后，即可生成真实图像。")
    
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
                # 确保在函数内可访问 openai 模块
                import openai
                
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