from data import load_data, Preprocessing
from model import transformer, CustomSchedule, accuracy, loss_function
from interface import ChatInterface
import tensorflow as tf

if __name__ == '__main__':
    data = load_data("csv")
    START_TOKEN, END_TOKEN, VOCAB_SIZE, tokenizer, dataset = Preprocessing(data)

    D_MODEL = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1

    model = transformer(vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, dff=DFF, d_model=D_MODEL, num_heads=NUM_HEADS, dropout=DROPOUT)

    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    EPOCHS = 30
    model.fit(dataset, epochs=EPOCHS)

    chat_interface = ChatInterface(model, START_TOKEN, END_TOKEN, tokenizer)
    chat_interface.get_interface().launch(share=True)

