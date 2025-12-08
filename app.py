import gradio as gr
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from dotenv import load_dotenv 

# ==================== 1. 从环境变量加载设置 ====================
load_dotenv() 

# 从环境变量读取，如果不存在则使用空字符串（防止报错）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")  # 使用兼容的默认模型

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
        # 为演示，仍使用占位图
        image_url = "https://via.placeholder.com/512x512/3A86FF/FFFFFF?text=Generated+Poster+Here"

        # 4. 返回给Gradio显示
        return decomposed_text, optimized_prompt, final_prompt, image_url

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