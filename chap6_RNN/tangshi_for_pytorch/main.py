import numpy as np
import collections
import torch
from sympy.physics.units import temperature
from torch.autograd import Variable
import torch.optim as optim

import rnn

start_token = 'G'
end_token = 'E'
batch_size = 64


# 检测是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def process_poems1(file_name):
    """

    :param file_name:
    :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
    e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

    """
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                # content = content.replace(' ', '').replace('，','').replace('。','')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                print("error")
                pass
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
        # content = ''
        for line in f.readlines():
            try:
                line = line.strip()
                if line:
                    content = line.replace(' '' ', '').replace('，','').replace('。','')
                    if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                    start_token in content or end_token in content:
                        continue
                    if len(content) < 5 or len(content) > 80:
                        continue
                    # print(content)
                    content = start_token + content + end_token
                    poems.append(content)
                    # content = ''
            except ValueError as e:
                # print("error")
                pass
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


def run_training():
    # 处理数据集
    # poems_vector, word_to_int, vocabularies = process_poems2('./tangshi.txt')
    poems_vector, word_to_int, vocabularies = process_poems1('./poems.txt')
    # 生成batch
    print("finish  loadding data")
    BATCH_SIZE = 100

    torch.manual_seed(5)
    word_embedding = rnn.word_embedding( vocab_length= len(word_to_int) + 1 , embedding_dim= 128) #改之前是100
    # rnn_model = rnn.RNN_model(batch_sz = BATCH_SIZE,vocab_len = len(word_to_int) + 1 ,word_embedding = word_embedding ,embedding_dim= 100, lstm_hidden_dim=128)
    # 修改模型初始化参数
    rnn_model = rnn.RNN_model(batch_sz=BATCH_SIZE,
                              vocab_len=len(word_to_int) + 1,
                              word_embedding=word_embedding,
                              embedding_dim=128,  # 增加嵌入维度
                              lstm_hidden_dim=256)  # 从128增加到256

    # 将模型移动到GPU
    word_embedding = word_embedding.to(device)
    rnn_model = rnn_model.to(device)

    optimizer = optim.Adam(rnn_model.parameters(), lr= 0.005)
    # optimizer=optim.RMSprop(rnn_model.parameters(), lr=0.01)

    loss_fun = torch.nn.NLLLoss()
    # rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))  # if you have already trained your model you can load it by this line.

    for epoch in range(100):    # 原来是30
        batches_inputs, batches_outputs = generate_batch(BATCH_SIZE, poems_vector, word_to_int)
        n_chunk = len(batches_inputs)
        for batch in range(n_chunk):
            batch_x = batches_inputs[batch]
            batch_y = batches_outputs[batch] # (batch , time_step)

            loss = 0
            for index in range(BATCH_SIZE):
                x = np.array(batch_x[index], dtype = np.int64)
                y = np.array(batch_y[index], dtype = np.int64)
                # x = Variable(torch.from_numpy(np.expand_dims(x,axis=1)))
                # y = Variable(torch.from_numpy(y ))
                x = torch.from_numpy(np.expand_dims(x, axis=1)).to(device)
                y = torch.from_numpy(y).to(device)
                x = Variable(x)
                y = Variable(y)
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
                torch.save(rnn_model.state_dict(), './poem_generator_rnn')
                print("finish  save model")



def to_word(predict, vocabs):  # 预测的结果转化成汉字
    sample = np.argmax(predict)
    # temperature = 1.0
    # # 应用温度参数调整概率分布
    # predict = np.array(predict) / temperature
    # # 转换为概率分布
    # probs = np.exp(predict) / np.sum(np.exp(predict))
    # # 按概率采样而不是取最大值
    # sample = np.random.choice(len(probs), p=probs)


    if sample >= len(vocabs):
        sample = len(vocabs) - 1

    return vocabs[sample]


def pretty_print_poem(poem):  # 令打印的结果更工整
    shige=[]
    for w in poem:
        if w == start_token or w == end_token:
            break
        shige.append(w)
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


def gen_poem(begin_word):
    # poems_vector, word_int_map, vocabularies = process_poems2('./tangshi.txt')  #  use the other dataset to train the network
    poems_vector, word_int_map, vocabularies = process_poems1('./poems.txt')
    word_embedding = rnn.word_embedding(vocab_length=len(word_int_map) + 1, embedding_dim=128)# 该之前为100
    # rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
    #                                embedding_dim=100, lstm_hidden_dim=128)
    rnn_model = rnn.RNN_model(batch_sz=64, vocab_len=len(word_int_map) + 1, word_embedding=word_embedding,
                              embedding_dim=128, lstm_hidden_dim=256)
    #rnn_model.load_state_dict(torch.load('./poem_generator_rnn'))
    ########################################gpu版本改的代码
    rnn_model.load_state_dict(torch.load('./poem_generator_rnn', map_location=device))
    rnn_model = rnn_model.to(device)
    ########################################
    # 指定开始的字

    poem = begin_word
    word = begin_word
    while word != end_token:
        ########################################gpu版本改之前的代码
        # input = np.array([word_int_map[w] for w in poem],dtype= np.int64)
        # input = Variable(torch.from_numpy(input))
        # output = rnn_model(input, is_test=True)
        # word = to_word(output.data.tolist()[-1], vocabularies)
        ########################################
        input = np.array([word_int_map[w] for w in poem], dtype=np.int64)
        input = torch.from_numpy(input).to(device)
        input = Variable(input)
        output = rnn_model(input, is_test=True)
        # 将输出移回CPU进行处理
        word = to_word(output.cpu().data.tolist()[-1], vocabularies)
        ########################################
        poem += word
        # print(word)
        # print(poem)
        if len(poem) > 30:
            break
    return poem


def save_poem_to_file(poem, begin_char, file_path='generated_poems.txt', mode='a'):
    """将生成的诗句保存到文件"""
    formatted_poem = []
    for w in poem:
        if w == start_token or w == end_token:
            continue  # 跳过开始和结束标记
        formatted_poem.append(w)

    poem_text = ''.join(formatted_poem)
    poem_sentences = poem_text.split('。')

    with open(file_path, mode, encoding='utf-8') as f:
        # 添加开头字标记
        f.write(f"【{begin_char}】\n")

        # 检查是否有内容可写入
        valid_sentences = [s for s in poem_sentences if s.strip()]

        if valid_sentences:
            for s in valid_sentences:
                if s.strip():  # 确保句子不是空字符串
                    f.write(s + '。\n')
        else:
            # 如果没有有效句子，至少输出一行原始内容
            f.write(poem_text + '\n')

        f.write('\n')  # 在不同诗之间添加空行



# run_training()  # 如果不是训练阶段 ，请注销这一行 。 网络训练时间很长。


def generate_and_save_multiple_poems(file_path='generated_poems.txt'):
    """生成多首诗并按格式保存到文件"""
    # 清空或创建文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("古诗生成集\n\n")

    # 要生成的诗的起始字列表
    start_chars = ["日", "红", "山", "夜", "湖", "海", "月", "君", "风", "云","张","力","文","高","乐","怡"]

    # 生成并保存每一首诗
    for char in start_chars:
        # 尝试最多3次生成有意义的诗
        for attempt in range(3):
            poem = gen_poem(char)
            if len(poem) > len(char) + 3:  # 确保生成了足够的内容
                break

        # 打印到控制台
        pretty_print_poem(poem)
        # 保存到文件
        save_poem_to_file(poem, char, file_path, 'a')

# pretty_print_poem(gen_poem("日"))
# pretty_print_poem(gen_poem("红"))
# pretty_print_poem(gen_poem("山"))
# pretty_print_poem(gen_poem("夜"))
# pretty_print_poem(gen_poem("湖"))
# pretty_print_poem(gen_poem("海"))
# pretty_print_poem(gen_poem("月"))
# pretty_print_poem(gen_poem("君"))

generate_and_save_multiple_poems()




