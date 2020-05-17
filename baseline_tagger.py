import hw2_corpus_tool as hw2
import pycrfsuite
import os
import sys
from itertools import zip_longest
import argparse

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

train_file = os.path.join(THIS_FOLDER, sys.argv[1])
dev_file = os.path.join(THIS_FOLDER, sys.argv[2])

train_files = list(hw2.get_data(train_file))
dev_files = list(hw2.get_data(dev_file))


def word2features(last_utterance, utterance, curr_utter):
    all_features = []
    if curr_utter == 0:
        last_speaker = curr_speaker = utterance[1]
    else:
        last_speaker = last_utterance[1]
        curr_speaker = utterance[1]
    x = 0
    if not utterance[2]:
        all_features = ['NO_WORDS']
    else:
        while x < len(utterance[2]):
            if curr_utter == 0:
                all_features.extend(['first_utterance=%s' % 'FIRST_UTTERANCE',
                                     'words_feature=%s' % 'TOKEN_' + utterance[2][x][0],
                                     'pos_tag=%s' % 'POS_' + utterance[2][x][1]
                                     ])
            else:
                if last_speaker != curr_speaker:
                    all_features.extend(['speaker_change=%s' % 'SPEAKER_CHANGED',
                                         'words_feature=%s' % 'TOKEN_' + utterance[2][x][0],
                                         'pos_tag=%s' % 'POS_' + utterance[2][x][1]
                                         ])
                else:
                    all_features.extend(['words_feature=%s' % 'TOKEN_' + utterance[2][x][0],
                                         'pos_tag=%s' % 'POS_' + utterance[2][x][1]
                                         ])
            x += 1
    features = set(all_features)
    return list(features)


def get_features_for_training(inp):
    x = []
    y = []
    for num in range(len(inp)):
        for utter in range(len(inp[num])):
            if num == 0:
                x.append(word2features(inp[num][utter], inp[num][utter], utter))
            else:
                x.append(word2features(inp[num][utter - 1], inp[num][utter], utter))

    for num in range(len(inp)):
        for utter in range(len(inp[num])):
            y.append(inp[num][utter][0])
    return x, y


train = get_features_for_training(train_files)
X_train = train[0]
Y_train = train[1]


test_data = get_features_for_training(dev_files)
Y_test = test_data[1]
X_test = test_data[0]

trainer = pycrfsuite.Trainer(verbose=False)
trainer.append(X_train, Y_train)
trainer.set_params({
    'c1': 1.0,  # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 50,  # stop earlier
    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
    })
trainer.train('baseline_tagger.crfsuite')

tagger = pycrfsuite.Tagger()
tagger.open('baseline_tagger.crfsuite')

y_pred = tagger.tag(X_test)

# original code
# with open(sys.argv[3], 'w') as outfile:
#     for n in range(len(dev_files)):
#         for tag in range(len(dev_files[n])):
#             outfile.write(y_pred[tag])
#             outfile.write("\n")
#         outfile.write("\n")

# changed code
#############################################
with open(sys.argv[3], 'w') as outfile:
    t = 0
    for n in range(len(dev_files)):
        for tag in range(len(dev_files[n])):
            outfile.write(y_pred[t])
            t += 1
            outfile.write("\n")
        outfile.write("\n")
##############################################

# def accuracy():
#     count = 0
#     for i in range(len(y_pred)):
#         print(y_pred[i], Y_test[i])
#         if y_pred[i] == Y_test[i]:
#             count += 1
#     #acc = count/len(Y_test)
#     print(len(y_pred))
#     return count


