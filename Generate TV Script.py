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
    counts = Counter(lines)
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





Here is the full error:
KeyError                                  Traceback (most recent call last)
<ipython-input-16-400f71299159> in <module>()
      3 """
      4 # pre-process training data
----> 5 helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

/home/workspace/helper.py in preprocess_and_save_data(dataset_path, token_lookup, create_lookup_tables)
     35 
     36     vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
---> 37     int_text = [vocab_to_int[word] for word in text]
     38     pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))
     39 

/home/workspace/helper.py in <listcomp>(.0)
     35 
     36     vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
---> 37     int_text = [vocab_to_int[word] for word in text]
     38     pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))
     39 

KeyError: 'this'
