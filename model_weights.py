import numpy as np
import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Concatenate, Permute, SpatialDropout1D, RepeatVector, LSTM, Bidirectional, Multiply, Lambda, Dense, Dropout, Input,Flatten,Embedding
from keras.models import Model
import keras.backend as K

train=pd.read_csv('toxic_train.csv')
train=train.sample(frac=1)
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"] # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
y = train[list_classes].values

max_features = 100000
maxlen = 200

list_sentences = train['comment_text'].values

# Tokenize corpus
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences)

# Add padding
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)



class Attention:
    def __call__(self, inp, combine=True, return_attention=True):
        
        repeat_size = int(inp.shape[-1])
        
        # Map through 1 Layer MLP
        x_a = Dense(repeat_size, kernel_initializer = 'glorot_uniform', activation="tanh", name="tanh_mlp")(inp) 
        
        # Dot with word-level vector
        x_a = Dense(1, kernel_initializer = 'glorot_uniform', activation='linear', name="word-level_context")(x_a)
        x_a = Flatten()(x_a) # x_a is of shape (?,200,1), we flatten it to be (?,200)
        att_out = Activation('softmax')(x_a) 
        
        # Clever trick to do elementwise multiplication of alpha_t with the correct h_t:
        # RepeatVector will blow it out to be (?,120, 200)
        # Then, Permute will swap it to (?,200,120) where each row (?,k,120) is a copy of a_t[k]
        # Then, Multiply performs elementwise multiplication to apply the same a_t to each
        # dimension of the respective word vector
        x_a2 = RepeatVector(repeat_size)(att_out)
        x_a2 = Permute([2,1])(x_a2)
        out = Multiply()([inp,x_a2])
        
        if combine:
        # Now we sum over the resulting word representations
            out = Lambda(lambda x : K.sum(x, axis=1), name='expectation_over_words')(out)
        
        if return_attention:
            out = (out, att_out)
                   
        return out
        
        
lstm_shape = 60
embed_size = 128

# Model details
inp = Input(shape=(maxlen,))
emb = Embedding(input_dim=max_features, input_length = maxlen, output_dim=embed_size)(inp)
x = SpatialDropout1D(0.35)(emb)
x = Bidirectional(LSTM(lstm_shape, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(x)
x, attention = Attention()(x)
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

attention_model = Model(inputs=inp, outputs=attention)

model.fit(X_t, y, validation_split=.2, epochs=3, verbose=1, batch_size=512)


# Helper functions
def get_reverse_token_map(tokenizer):
    reverse_token_map = dict(map(reversed, tokenizer.word_index.items()))
    return reverse_token_map

def get_word_importances(text):
    reverse_token_map = get_reverse_token_map(tokenizer)
    lt = tokenizer.texts_to_sequences([text])
    x = pad_sequences(lt, maxlen=maxlen)
    p = model.predict(x)
    att = attention_model.predict(x)
    return p, [(reverse_token_map.get(word), importance) for word, importance in zip(x[0], att[0]) if word in reverse_token_map]
    
    
### EXAMPLES
get_word_importances('She prefers uplifting movies') # ..., [('she', 0.06030082), ('prefers', 0.08760931), ('movies', 0.10084405)])
get_word_importances('You are an asshole') # ..., [('you', 0.014374573), ('are', 0.023701496), ('an', 0.02919567), ('asshole', 0.9242637)])
