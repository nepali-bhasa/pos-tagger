import os
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download("indian")

from nltk.tag import tnt
from nltk.corpus import indian


def nepali_model():
    data_path = os.path.join(os.getcwd(), 'data/nepali.pos')
    train_data = indian.tagged_sents(data_path)
    tnt_pos_tagger = tnt.TnT()
    tnt_pos_tagger.train(train_data)
    return tnt_pos_tagger


def pos_tag(text: str) -> list:
    model = nepali_model()
    return model.tag(nltk.word_tokenize(text))


if __name__ == "__main__":
    print(pos_tag('मंगलबार बिहानसम्म इरानमा कोरोना संक्रमणबाट ८ सय ५३ जनाको मृत्यु भएको छ भने '
                  '१४ हजार ९ सय ९१ जना संक्रमित भएका छन् ।'))
