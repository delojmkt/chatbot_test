from model.prediction import predict
import gradio as gr

__all__=["ChatInterface"]

class ChatInterface:
    def __init__(self, model, START_TOKEN, END_TOKEN, tokenizer):
        self.model = model
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.tokenizer = tokenizer

    def response(self, message, history, additional_input_info):
        return predict(message, self.model, self.START_TOKEN, self.END_TOKEN, self.tokenizer)
    
    def get_interface(self):
        return gr.ChatInterface(
        fn=self.response,
        textbox=gr.Textbox(placeholder="말걸어주세요..", container=False, scale=7),
        # 채팅창의 크기를 조절한다.
        chatbot=gr.Chatbot(),
        title="김다은 테스트용 챗봇 입니다.",
        description="아직 충분한 데이터로 훈련을 끝내지 못해 성능이 다소 떨어집니다.",
        theme="soft",
        examples=[["12시 땡!"], ["SD카드 망가졌어"], ["결정을 못 내리겠어. 어떻해"]],
        retry_btn="다시보내기 ↩",
        undo_btn="이전챗 삭제 ❌",
        clear_btn="전챗 삭제 💫")