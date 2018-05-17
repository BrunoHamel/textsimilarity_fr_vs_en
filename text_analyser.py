import string
from statistics import mean
from typing import Callable, List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


class TextAnalyser:
    """
    >>> t.compare('Hello', 'Bonjour', t.jaccard_similarity)
    1.0
    """

    def _lem_tokenize(self, text: str) -> List:
        """
        Tokenizes, lemmatize and removes punctuations from a string

        Args:
            text (str): The text to process.

        Returns:
           list: Tokens outputed from processes.

        >>> t._lem_tokenize('many cars are on the roads.')
        ['many', 'car', 'are', 'on', 'the', 'road']
        """

        lemmer = nltk.stem.WordNetLemmatizer()
        table = str.maketrans({key: None for key in string.punctuation})
        text = text.translate(table)
        tokens = ToktokTokenizer().tokenize(text)

        tokenized = [lemmer.lemmatize(token) for token in tokens]

        return tokenized

    def _translate_to(self, text: str, lang: str = 'en') -> str:
        """
        Translates string to a language.

        Args:
            text (str): Text to translate.
            lang (str): language to translate into. Defaults to 'en'.

        Returns:
           str: Translated text.

        >>> t._translate_to('Bonjour, je suis un programme')
        'Hello, I am a program'

        >>> t._translate_to('Hello, I am a program', 'fr')
        'Bonjour, je suis un programme'

        """
        return str(TextBlob(text).translate(to=lang))

    def _pre_process(self, text: str, lang: str='en') -> str:
        """
        Pre process string before yb removing stop words and lowering words.

        Args:
            text (str): Text to process.
            lang (str): Language of the text. Defaults to 'en'.

        Returns:
           str: Processed text.

        >>> t._pre_process('This is for testing')
        'this testing'
        """

        if lang == 'en':
            lang = 'english'
        else:
            lang = 'french'

        blob = TextBlob(text)
        words = [word.lower()
                 for word in blob.words if word not in stopwords.words(lang)]

        return ' '.join(words)

    def cos_similarity(self, text_fr: str, text_en: str) -> float:
        """
        Cosine similarity calculates the cosinus angle
        between the two vector of words.

        Args:
            text_en (str): English text.
            text_fr (str): French text.

        Returns:
           float: Cosine similarity.

        >>> t.cos_similarity('Hello', 'Hello')
        1.0
        """
        text_list = [text_fr, text_en]

        TfidfVec = TfidfVectorizer(tokenizer=self._lem_tokenize)

        tfidf = TfidfVec.fit_transform(text_list)
        return (tfidf * tfidf.T).toarray()[0][1]

    def jaccard_similarity(self, text_fr: str, text_en: str) -> float:
        """
        Jaccard similarity is a statistic used for comparing
        the similarity and diversity of sample sets. It measures
        similarity between finite sample sets, and is defined
        as the size of the intersection divided by the size of
        the union of the sample sets.

        Args:
            text_en (str): English text.
            text_fr (str): French text.

        Returns:
           float: Jaccard similarity.

        >>> t.jaccard_similarity('Hello', 'Hello')
        1.0
        """

        a = set(text_fr.split())
        b = set(text_en.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))

    def levenshtein_similarity(self, text_fr: str, text_en: str) -> float:
        """
        Levenshtein calculates the number of changes
        that are needed to go from a string to another.

        Args:
            text_en (str): English text.
            text_fr (str): French text.

        Returns:
           float: 1 - (number of changes / length of string).

        >>> t.levenshtein_similarity("Programming is cool", "Programming is awsome")
        0.75
        """
        if len(text_fr) < len(text_en):
            return self.levenshtein(text_en, text_fr)

        if len(text_en) == 0:
            return len(text_fr)

        previous_row = range(len(text_en) + 1)
        for i, c1 in enumerate(text_fr):
            current_row = [i + 1]
            for j, c2 in enumerate(text_en):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return 1 - (previous_row[-1] / mean([len(text_fr), len(text_en)]))

    def compare(self, text_en: str, text_fr: str,
                similarity_metric: Callable) -> float:
        """
        Compare two string with a similarity algorithm.

        Args:
            text_en (str): English text.
            text_fr (str): French text.
            similarity_metric (Callable): Similarity algorithm.

        Returns:
           float: The similarity metric.

        >>> t.compare('Hello', 'Bonjour', t.cos_similarity)
        1.0
        """
        translated_fr = self._translate_to(text_fr, 'en')

        processed_en = self._pre_process(text_en)
        processed_fr = self._pre_process(translated_fr)

        sim = similarity_metric(processed_en, processed_fr)

        return sim

    def is_same(self, text_en: str, text_fr: str,
                threshold: float=0.29) -> float:
        """
        Check if two strings are the same.

        Args:
            text_en (str): English text.
            text_fr (str): French text.
            threshold (float): Cosine similarity value considered high enough.

        Returns:
            bool: If the two strings are the same according to the threshold.

        >>> t.is_same('Hello', 'Bonjour')
        True
        """
        return self.compare(text_en, text_fr, self.cos_similarity) > threshold


if __name__ == "__main__":
    import doctest
    doctest.testmod(extraglobs={'t': TextAnalyser()})
