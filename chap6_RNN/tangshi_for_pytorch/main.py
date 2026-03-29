import os
import numpy as np
import collections
import torch
from torch.autograd import Variable
import torch.optim as optim

import rnn

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
poems_file = os.path.join(script_dir, 'poems.txt')
model_file = os.path.join(script_dir, 'poem_generator_rnn')

start_token = 'G'
end_token = 'E'
batch_size = 64


def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            line = line.strip()
            if not line or ':' not in line:
                continue
            title, content = line.split(':', 1)
            # content = content.replace(' ', '').replace('，','').replace('。','')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                continue
            if len(content) < 5 or len(content) > 80:
                continue
            content = start_token + content + end_token
            poems.append(content)
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def process_poems2(file_name):
    """
    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            content = line.replace(' ', '').replace('，','').replace('。','')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                            start_token in content or end_token in content:
                continue
            if len(content) < 5 or len(content) > 80:
                continue
            content = start_token + content + end_token
            poems.append(content)
    # 按诗的字数排序
    poems = sorted(poems, key=lambda line: len(line))
    # print(poems)
    # 统计每个字出现次数
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # 统计词和词频。
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words

def generate_batch(batch_size, poems_vec, word_to_int):
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size
        x_data = poems_vec[start_index:end_index]
        y_data = []
        for row in x_data:
            y  = row[1:]
            y.append(row[-1])
            y_data.append(y)
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        # print(x_data[0])
        # print(y_data[0])
        # exit(0)
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def build_model(vocab_size, batch_size):
    word_embedding = rnn.word_embedding(vocab_length=vocab_size, embedding_dim=100)
    return rnn.RNN_model(
        batch_sz=batch_size,
        vocab_len=vocab_size,
        word_embedding=word_embedding,
        embedding_dim=100,
        lstm_hidden_dim=128,
    )


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1(poems_file)
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    rnn_model = build_model(vocab_size=len(word_to_int) + 1, batch_size=BATCH_SIZE)

    # optimizer = optim.Adam(rnn_model.parameters(), lr= 0.001)
    optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(30):
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))
                y = Variable(torch.from_numpy(y ))
                pre = rnn_model(x)
                loss += loss_fun(pre , y)
                if index == 0:
                    _, pre = torch.max(pre, dim=1)
                    print('prediction', pre.data.tolist()) # the following  three line can print the output and the prediction
                    print('b_y       ', y.data.tolist())   # And you need to take a screenshot and then past is to your homework paper.
                    print('*' * 30)
            loss  = loss  / BATCH_SIZE
            print("epoch  ",epoch,'batch number',batch,"loss is: ", loss.data.tolist())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn_model.parameters(), 1)
            optimizer.step()

            if batch % 20 ==0:
                torch.save(rnn_model.state_dict(), model_file)
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)

    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    poem = poem.replace(start_token, '').split(end_token)[0]
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def load_inference_resources():
    _, word_int_map, vocabularies = process_poems1(poems_file)
    rnn_model = build_model(vocab_size=len(word_int_map) + 1, batch_size=1)
    rnn_model.load_state_dict(torch.load(model_file, map_location='cpu'))
    rnn_model.eval()
    return rnn_model, word_int_map, vocabularies


def gen_poem(begin_word, rnn_model, word_int_map, vocabularies):
    if begin_word not in word_int_map:
        raise ValueError("begin_word not in vocabulary")

    poem = begin_word
    word = begin_word
    while word != end_token:
        input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        input = Variable(torch.from_numpy(input))
        output = rnn_model(input, is_test=True)
        word = to_word(output.data.tolist()[-1], vocabularies)
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem


if __name__ == "__main__":
    # run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。

    infer_model, infer_word_int_map, infer_vocabularies = load_inference_resources()
    for begin_word in ["日", "红", "山", "夜", "湖", "海", "月"]:
        pretty_print_poem(gen_poem(begin_word, infer_model, infer_word_int_map, infer_vocabularies))
