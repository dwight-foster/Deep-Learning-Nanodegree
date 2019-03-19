"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# load in data
import helper
data_dir = './data/Seinfeld_Scripts.txt'
text = helper.load_data(data_dir)


view_line_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))

lines = text.split('\n')
print('Number of lines: {}'.format(len(lines)))
word_count_line = [len(line.split()) for line in lines]
print('Average number of words in each line: {}'.format(np.average(word_count_line)))

print()
print('The lines {} to {}:'.format(*view_line_range))
print('\n'.join(text.split('\n')[view_line_range[0]:view_line_range[1]]))


import problem_unittests as tests
from collections import Counter
def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    int_to_vocab = {ii: word for ii, word in enumerate(vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
   
    # return tuple
    return (vocab_to_int, int_to_vocab)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)



def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    dict = {'.': '||Period||', ',': '||Comma||', '"': '||QuotationMark||', ';': '||Semicolon||', '!' : '||Exclamationmark||', 
           '?': '||Questionmark||', '(': '||LeftParentheses||', ')': '||RightParentheses||', '-': '||Dash||', '\n':'||Return||'}    
    return dict

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)



"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# pre-process training data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)




Here is the batch code:
from torch.utils.data import TensorDataset, DataLoader
import torch

def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    n_batches = int(len(words)/(batch_size * sequence_length))
    xdata = np.array(words[: n_batches * batch_size * sequence_length])
    ydata = np.array(words[:batch_size ])
    
    x_batches = np.array(np.split(xdata.reshape(-1,batch_size),n_batches))
    y_batches = np.array(np.split(ydata.reshape(batch_size),n_batches))
    
    x_data = TensorDataset(torch.from_numpy(x_batches), torch.from_numpy(y_batches))
    data = DataLoader(x_data, shuffle=True, batch_size=batch_size)
        # return a dataloader
    return data

# there is no test for this function, but you are encouraged to create
# print statements and tests of your own
