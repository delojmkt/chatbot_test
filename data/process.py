import re
import tensorflow_datasets as tfds
import tensorflow as tf

__all__=["Preprocessing"]

class Processing:

    def __init__(self) -> None:
        self.questions = []
        self.answers = []
        self.tokenizer = None

    def _get_vocab_set(self, data):
        for sentence in data['Q']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            self.questions.append(sentence)

        for sentence in data['A']:
            sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
            sentence = sentence.strip()
            self.answers.append(sentence)

        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
        self.questions + self.answers, target_vocab_size=2**13)
        
    def get_token(self, data):
        self._get_vocab_set(data)

        START_TOKEN, END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        VOCAB_SIZE = self.tokenizer.vocab_size + 2

        return START_TOKEN, END_TOKEN, VOCAB_SIZE, self.tokenizer
    
    def tokenize_and_filter(self, START_TOKEN, END_TOKEN, MAX_LENGTH:int=40):
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(self.questions, self.answers):
            # encode(토큰화 + 정수 인코딩), 시작 토큰과 종료 토큰 추가
            sentence1 = START_TOKEN + self.tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + self.tokenizer.encode(sentence2) + END_TOKEN

            tokenized_inputs.append(sentence1)
            tokenized_outputs.append(sentence2)

        # 패딩
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs
    
def Preprocessing(data, BATCH_SIZE:int=64, BUFFER_SIZE:int=20000):
    preprocessing = Processing()
    START_TOKEN, END_TOKEN, VOCAB_SIZE, tokenizer = preprocessing.get_token(data)
    questions, answers = preprocessing.tokenize_and_filter(START_TOKEN, END_TOKEN)

    dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1] # 디코더의 입력 / 마지막 패딩 토큰 제거
    },
    {
        'outputs': answers[:, 1:]  # 맨 처음 토큰이 제거 = 시작 토큰 제거
    },
))

    dataset = dataset.cache()
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return START_TOKEN, END_TOKEN, VOCAB_SIZE, tokenizer, dataset
    
def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence