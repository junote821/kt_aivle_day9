# ============================================================
# 0) 환경 설정과 라이브러리 불러오기
# ------------------------------------------------------------
# - dotenv: .env 파일에서 OPENAI_API_KEY 같은 민감한 값을 읽어옵니다.
# - openai: OpenAI API를 쓰기 위한 공식 SDK입니다.
# - asyncio: 비동기 함수를 실행하기 위한 파이썬 표준 라이브러리입니다.
# - base64: 이미지 등 바이너리 데이터를 텍스트 형태로 바꾸는 데 씁니다.
# - streamlit: 웹앱(채팅 UI)을 쉽게 만들 수 있는 라이브러리입니다.
# - agents: 본 예제에서 사용하는 "에이전트 프레임워크" 관련 클래스/도구들입니다.
# ============================================================
import dotenv
dotenv.load_dotenv()  # .env 파일에서 환경 변수를 읽어 현재 프로세스에 주입

from openai import OpenAI
import asyncio
import base64
import streamlit as st

from agents import (
    Agent,               # 대화 에이전트를 생성하는 클래스
    Runner,              # 에이전트를 실행(스트리밍)하는 도우미
    SQLiteSession,       # 대화 히스토리를 로컬 SQLite DB에 저장/읽기
    WebSearchTool,       # 웹 검색 도구
    FileSearchTool,      # 벡터 스토어 기반 파일 검색 도구
    ImageGenerationTool, # 이미지 생성 도구
    CodeInterpreterTool, # 코드 실행 도구(샌드박스)
    HostedMCPTool,       # 외부 MCP 서버에 연결해 도구처럼 쓰는 기능
)

# OpenAI 클라이언트(파일 업로드, 벡터 스토어 관리 등에서 사용)
client = OpenAI()


# ============================================================
# 1) "벡터 스토어" 준비 함수
# ------------------------------------------------------------
# 벡터 스토어란?
# - 업로드한 파일(텍스트/이미지 등)을 "검색이 잘 되도록" 인덱싱 해두는 저장소입니다.
# - 나중에 "이 파일 내용으로 답해줘" 같은 요청을 할 때 RAG(검색+생성)에 활용됩니다.
#
# 이 함수는 다음을 해줍니다:
#  - (1) 이미 세션에 저장된 벡터 스토어 ID가 있으면 그게 정말 존재하는지 확인
#  - (2) 없거나, 과거 하드코딩 ID가 현재 프로젝트에 존재하지 않으면 새로 만듦
#  - 만들어진 ID는 st.session_state["VECTOR_STORE_ID"]에 저장 → 새로고침 전까지 재사용
# ============================================================
VECTOR_STORE_NAME = "chatgpt-clone-store"  # 사람이 보기 쉬운 이름(표시용)
# 과거에 쓰던(혹은 문서에 적혀있던) 벡터 스토어 ID. 현재 프로젝트/조직과 다르면 없을 수 있음.
DEFAULT_VECTOR_STORE_ID = "vs_68a0815f62388191a9c3701ceb237234"

def ensure_vector_store() -> str:
    """세션에 유효한 벡터 스토어 ID를 확보해 반환합니다. 없으면 새로 생성합니다."""
    # 1) 이미 세션에 저장된 ID가 있는지 확인
    vs_id = st.session_state.get("VECTOR_STORE_ID")

    # 2) 후보군(candidates) 순회하며 실제 존재하는지 확인
    candidates = [vs_id, DEFAULT_VECTOR_STORE_ID]
    for cand in candidates:
        if cand:
            try:
                client.vector_stores.retrieve(vector_store_id=cand)
                # 문제 없이 조회되면 사용 가능 → 세션에도 보관
                st.session_state["VECTOR_STORE_ID"] = cand
                return cand
            except Exception:
                # 조회 실패(없음/권한없음/다른 프로젝트 ID 등)는 무시하고 다음 후보 시도
                pass

    # 3) 여기까지 왔다면 쓸만한 스토어가 없음 → 새로 만든다
    vs = client.vector_stores.create(name=VECTOR_STORE_NAME)
    st.session_state["VECTOR_STORE_ID"] = vs.id
    return vs.id

# 앱 시작 시점에 한 번 보장해 두면, 이후 코드에서 편하게 가져다 씁니다.
VECTOR_STORE_ID = ensure_vector_store()


# ============================================================
# 2) 에이전트(보조 모델) 만들기
# ------------------------------------------------------------
# - name: 에이전트 이름(표시용)
# - instructions: 모델에게 줄 "역할/가이드" 설명
# - tools: 모델이 필요할 때 사용할 수 있는 도구 목록
#
# 여기서는 5가지 도구를 제공합니다:
#  - WebSearchTool: 최신 정보가 필요할 때 웹 검색
#  - FileSearchTool: 업로드한 파일(벡터 스토어 기반)에서 근거를 찾아 답변
#  - ImageGenerationTool: 그림/이미지 생성
#  - CodeInterpreterTool: 코드로 계산/분석/그래프 출력
#  - HostedMCPTool: 외부 MCP(Server) 도구(여기선 Context7)를 사용
#
# st.session_state에 넣는 이유:
# - Streamlit은 화면이 바뀔 때마다 코드가 위에서 아래로 다시 실행됩니다.
# - 같은 에이전트를 계속 쓰고 싶어서(매번 새로 만들지 않도록) 세션 상태에 저장합니다.
# ============================================================
if "agent" not in st.session_state:
    st.session_state["agent"] = Agent(
        name="ChatGPT Clone",
        instructions="""
        You are a helpful assistant.

        You have access to the followign tools:
            - Web Search Tool: Use this when the user asks a questions that isn't in your training data. Use this tool when the users asks about current or future events, when you think you don't know the answer, try searching for it in the web first.
            - File Search Tool: Use this tool when the user asks a question about facts related to themselves. Or when they ask questions about specific files.
            - Code Interpreter Tool: Use this tool when you need to write and run code to answer the user's question.
        """,
        tools=[
            WebSearchTool(),
            FileSearchTool(
                vector_store_ids=[VECTOR_STORE_ID],  # 위에서 보장한 벡터 스토어를 연결
                max_num_results=3,                   # 파일 검색 결과 최대 3개 정도로 제한
            ),
            ImageGenerationTool(
                tool_config={
                    "type": "image_generation",
                    "quality": "high",
                    "output_format": "jpeg",
                    "partial_images": 1,  # 생성 중간 프리뷰를 받을지 여부
                }
            ),
            CodeInterpreterTool(
                tool_config={
                    "type": "code_interpreter",
                    "container": {
                        "type": "auto",   # 도커/샌드박스 컨테이너 자동 선택
                    },
                }
            ),
            HostedMCPTool(
                tool_config={
                    "server_url": "https://mcp.context7.com/mcp",
                    "type": "mcp",
                    "server_label": "Context7",
                    "server_description": "Use this to get the docs from software projects.",
                    "require_approval": "never",  # 매 호출 승인 팝업 없이 바로 실행
                }
            ),
        ],
    )
agent = st.session_state["agent"]  # 편의 변수로 가져오기


# ============================================================
# 3) 대화 히스토리 저장소(세션) 준비
# ------------------------------------------------------------
# SQLiteSession:
#  - 로컬 파일 DB(chat-gpt-clone-memory.db)에 대화 내용을 저장합니다.
#  - 앱이 새로고침되어도 이전 대화 기록을 쉽게 복원할 수 있습니다.
# ============================================================
if "session" not in st.session_state:
    st.session_state["session"] = SQLiteSession(
        "chat-history",                 # 대화 "채널" 이름(표)
        "chat-gpt-clone-memory.db",     # 저장될 로컬 DB 파일명
    )
session = st.session_state["session"]


# ============================================================
# 4) 과거 대화(히스토리) 화면에 그리기
# ------------------------------------------------------------
# - Streamlit은 "함수 실행 결과를 즉시 그리는" 방식입니다.
# - 비동기 함수로 DB에서 메시지를 읽고, 말풍선 형태로 차곡차곡 내보냅니다.
# - 사용자 메시지(텍스트/이미지)와, 에이전트 메시지(텍스트/도구호출 로그)를 구분해서 표시합니다.
# ============================================================
async def paint_history():
    messages = await session.get_items()  # 이전에 저장된 메시지들 읽기(리스트)

    for message in messages:
        # 4-1) user/assistant 같은 역할(role)이 있는 일반 메시지
        if "role" in message:
            # 역할에 맞는 말풍선을 생성
            with st.chat_message(message["role"]):
                if message["role"] == "user":
                    # 사용자의 입력은 문자열 또는 이미지 파트 리스트일 수 있습니다.
                    content = message["content"]
                    if isinstance(content, str):
                        st.write(content)  # 텍스트 그대로 출력
                    elif isinstance(content, list):
                        # 이미지가 들어있다면 렌더링
                        for part in content:
                            if "image_url" in part:
                                st.image(part["image_url"])
                else:
                    # 에이전트(assistant) 메시지
                    if message.get("type") == "message":
                        # 수식 표시 문제를 피하려고 '$'를 '\$' 로 바꿉니다. (기존 코드 유지)
                        st.write(message["content"][0]["text"].replace("$", "\$"))

        # 4-2) 도구 호출 로그(웹검색/코드실행/이미지생성 등)는 type으로 구분됩니다.
        if "type" in message:
            message_type = message["type"]
            if message_type == "web_search_call":
                with st.chat_message("ai"):
                    st.write("🔍 Searched the web...")
            elif message_type == "file_search_call":
                with st.chat_message("ai"):
                    st.write("🗂️ Searched your files...")
            elif message_type == "image_generation_call":
                # 이미지 생성 결과는 base64로 오므로 디코딩해서 보여줍니다.
                image = base64.b64decode(message["result"])
                with st.chat_message("ai"):
                    st.image(image)
            elif message_type == "code_interpreter_call":
                with st.chat_message("ai"):
                    st.code(message["code"])  # 실행했던 코드 보여주기
            elif message_type == "mcp_list_tools":
                with st.chat_message("ai"):
                    # 따옴표 문제를 피하기 위해 [] 안에는 작은따옴표를 씁니다.
                    st.write(f"Listed {message['server_label']}'s tools")
            elif message_type == "mcp_call":
                with st.chat_message("ai"):
                    st.write(
                        f"Called {message['server_label']}'s {message['name']} with args {message['arguments']}"
                    )

# 비동기 함수를 즉시 실행(현재 구조는 asyncio.run 유지)
asyncio.run(paint_history())


# ============================================================
# 5) 상태(프로그램이 무엇을 하는지) 표시 도우미
# ------------------------------------------------------------
# - 모델이 "웹검색 시작/완료", "코드 실행 중/완료" 같은 이벤트를 스트리밍으로 보내면
#   여기서 사람이 읽기 쉬운 라벨과 상태(running/complete)를 정해서 표시합니다.
# - st.status(...) 컴포넌트의 label/state를 업데이트합니다.
# ============================================================
def update_status(status_container, event):
    status_messages = {
        "response.web_search_call.completed": ("✅ Web search completed.", "complete"),
        "response.web_search_call.in_progress": ("🔍 Starting web search...", "running"),
        "response.web_search_call.searching": ("🔍 Web search in progress...", "running"),
        "response.file_search_call.completed": ("✅ File search completed.", "complete"),
        "response.file_search_call.in_progress": ("🗂️ Starting file search...", "running"),
        "response.file_search_call.searching": ("🗂️ File search in progress...", "running"),
        "response.image_generation_call.generating": ("🎨 Drawing image...", "running"),
        "response.image_generation_call.in_progress": ("🎨 Drawing image...", "running"),
        "response.code_interpreter_call_code.done": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.completed": ("🤖 Ran code.", "complete"),
        "response.code_interpreter_call.in_progress": ("🤖 Running code...", "complete"),
        "response.code_interpreter_call.interpreting": ("🤖 Running code...", "complete"),
        "response.mcp_call.completed": ("⚒️ Called MCP tool", "complete"),
        "response.mcp_call.failed": ("⚒️ Error calling MCP tool", "complete"),
        "response.mcp_call.in_progress": ("⚒️ Calling MCP tool...", "running"),
        "response.mcp_list_tools.completed": ("⚒️ Listed MCP tools", "complete"),
        "response.mcp_list_tools.failed": ("⚒️ Error listing MCP tools", "complete"),
        "response.mcp_list_tools.in_progress": ("⚒️ Listing MCP tools", "running"),
        "response.completed": (" ", "complete"),
    }
    if event in status_messages:
        label, state = status_messages[event]
        status_container.update(label=label, state=state)


# ============================================================
# 6) 에이전트 한 턴 실행(스트리밍)
# ------------------------------------------------------------
# - 사용자가 입력한 메시지를 전달하면, Runner.run_streamed(...)가 스트림 이벤트를 보냅니다.
#   (텍스트 토큰이 조금씩 늘어나거나, 도구가 호출되거나, 이미지가 도착하는 등)
# - 우리는 이 이벤트를 받으면서 화면의 플레이스홀더를 "실시간 업데이트" 합니다.
#   (text_placeholder, code_placeholder, image_placeholder)
# ============================================================
async def run_agent(message):
    with st.chat_message("ai"):
        # 왼쪽 말풍선 안에 "상태 영역"과 "텍스트/코드/이미지" 자리(플레이스홀더)를 깔아둡니다.
        status_container = st.status("⏳", expanded=False)
        code_placeholder = st.empty()
        image_placeholder = st.empty()
        text_placeholder = st.empty()

        # 스트림으로 붙을 텍스트/코드를 누적할 변수
        response = ""
        code_response = ""

        # 혹시 다른 함수에서 접근할 수 있도록 세션에 저장
        st.session_state["code_placeholder"] = code_placeholder
        st.session_state["image_placeholder"] = image_placeholder
        st.session_state["text_placeholder"] = text_placeholder

        # 벡터 스토어가 혹시 사라졌거나(다른 탭에서 초기화) 바뀌었을 수 있으니 실행 직전에 한 번 더 보장
        vs_id = ensure_vector_store()

        # 에이전트를 스트리밍 모드로 실행
        stream = Runner.run_streamed(
            agent,
            message,
            session=session,  # 대화 기록을 이 세션 DB에 저장
        )

        # 모델이 보내는 다양한 이벤트를 순서대로 처리
        async for event in stream.stream_events():
            if event.type == "raw_response_event":
                # 상태 라벨 업데이트(지금 무슨 단계인지)
                update_status(status_container, event.data.type)

                # 6-1) 텍스트가 토큰 단위로 도착할 때
                if event.data.type == "response.output_text.delta":
                    response += event.data.delta
                    # '$' → '\$' 치환(기존 코드 유지)
                    text_placeholder.write(response.replace("$", "\$"))

                # 6-2) 코드 인터프리터가 "실행할 코드"를 스트림으로 흘려보낼 때
                if event.data.type == "response.code_interpreter_call_code.delta":
                    code_response += event.data.delta
                    code_placeholder.code(code_response)

                # 6-3) 이미지 생성 도중 "부분 이미지"가 올 때(프리뷰)
                elif event.data.type == "response.image_generation_call.partial_image":
                    image = base64.b64decode(event.data.partial_image_b64)
                    image_placeholder.image(image)


# ============================================================
# 7) 입력창(채팅 UI) + 파일 업로드 처리
# ------------------------------------------------------------
# - 사용자가 아래 입력 상자에 텍스트를 쓰거나, 파일을 끌어다 놓을 수 있습니다.
# - 파일을 올리면 OpenAI Files & Vector Store에 업로드하고 연결합니다.
# - 텍스트가 있으면 run_agent(...)를 실행해서 답변을 받습니다.
# ============================================================
prompt = st.chat_input(
    "Write a message for your assistant",
    accept_file=True,                 # 파일 업로드 허용
    file_type=["txt", "jpg", "jpeg", "png"],  # 허용 파일 형식
)

if prompt:

    # 새 요청이 들어오면 이전 플레이스홀더 비우기(화면 깔끔)
    if "code_placeholder" in st.session_state:
        st.session_state["code_placeholder"].empty()
    if "image_placeholder" in st.session_state:
        st.session_state["image_placeholder"].empty()
    if "text_placeholder" in st.session_state:
        st.session_state["text_placeholder"].empty()

    # 업로드된 파일이 있으면 먼저 처리
    for file in prompt.files:
        if file.type.startswith("text/"):
            # 텍스트 파일은 OpenAI Files로 올린 뒤, 벡터 스토어에 연결하여
            # 나중에 FileSearchTool이 검색할 수 있게 합니다.
            with st.chat_message("ai"):
                with st.status("⏳ Uploading file...") as status:
                    vs_id = ensure_vector_store()  # 안전하게 다시 보장

                    # (1) 원본 파일 업로드 (user_data 용도)
                    uploaded_file = client.files.create(
                        file=(file.name, file.getvalue()),
                        purpose="user_data",
                    )
                    status.update(label="⏳ Attaching file...")

                    # (2) 벡터 스토어와 파일 연결 → 인덱싱되어 검색 가능
                    client.vector_stores.files.create(
                        vector_store_id=vs_id,
                        file_id=uploaded_file.id,
                    )
                    status.update(label="✅ File uploaded", state="complete")

        elif file.type.startswith("image/"):
            # 이미지 파일은 대화 히스토리에 "사용자 이미지"로 직접 추가합니다.
            with st.status("⏳ Uploading image...") as status:
                file_bytes = file.getvalue()
                # base64로 변환하여 data URI 형식으로 저장
                base64_data = base64.b64encode(file_bytes).decode("utf-8")
                data_uri = f"data:{file.type};base64,{base64_data}"

                # 비동기 DB API를 현재 구조에서는 asyncio.run으로 즉시 실행
                asyncio.run(
                    session.add_items(
                        [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_image",
                                        "detail": "auto",
                                        "image_url": data_uri,
                                    }
                                ],
                            }
                        ]
                    )
                )
                status.update(label="✅ Image uploaded", state="complete")

            # 업로드 완료 후, 사용자 말풍선으로 미리보기
            with st.chat_message("human"):
                st.image(data_uri)

    # 텍스트가 있으면 실제로 에이전트를 돌립니다.
    if prompt.text:
        with st.chat_message("human"):
            st.write(prompt.text)  # 사용자 입력 에코
        asyncio.run(run_agent(prompt.text))  # 한 턴 실행


# ============================================================
# 8) 사이드바(보조 기능)
# ------------------------------------------------------------
# - Reset memory: 현재 SQLiteSession에 저장된 대화 기록을 삭제
# - 현재 세션 히스토리 보기: 디버깅/학습용으로 내부 저장 데이터를 보여줌
# ============================================================
with st.sidebar:
    reset = st.button("Reset memory")
    if reset:
        asyncio.run(session.clear_session())  # 모든 히스토리 삭제

    # 현재 저장된 히스토리를 그대로 출력(학습/디버그용)
    st.write(asyncio.run(session.get_items()))
