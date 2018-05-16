import string

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


class TextAnalyser:
    def _lem_normalize(self, text):
        _remove_punct_dict = dict((ord(punct), None)
                                  for punct in string.punctuation)

        return self._lem_tokens(nltk.word_tokenize(
            text.translate(_remove_punct_dict)))

    def _lem_tokens(self, tokens):
        lemmer = nltk.stem.WordNetLemmatizer()
        return [lemmer.lemmatize(token) for token in tokens]

    def _translate_to(self, text, lang='en'):
        return str(TextBlob(text).translate(to=lang))

    def _pre_process(self, text, lang='en'):
        if lang == 'en':
            lang = 'english'
        else:
            lang = 'french'

        blob = TextBlob(text)
        words = [word.lemmatize().lower()
                 for word in blob.words if word not in stopwords.words(lang)]

        return ' '.join(words)

    def cos_similarity(self, text_fr, text_en):
        text_list = [text_fr, text_en]

        TfidfVec = TfidfVectorizer(tokenizer=self._lem_normalize)

        tfidf = TfidfVec.fit_transform(text_list)
        return (tfidf * tfidf.T).toarray()[0][1]

    def jaccard_similarity(self, text_fr, text_en):
        a = set(text_fr.split())
        b = set(text_en.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def compare(self, text_en, text_fr, similarity_metric):
        translated_fr = self._translate_to(text_fr, 'en')

        processed_en = self._pre_process(text_en)
        processed_fr = self._pre_process(translated_fr)

        sim = similarity_metric(processed_en, processed_fr)

        return sim

    def is_same(self, text_en, text_fr, threshold=0.29):
        return self.compare(text_en, text_fr) > threshold
