import math
import re
import string

import nltk
import sys
import os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_content = dict()

    # Loop files in directory
    for file_name in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:
        # Read file content
        with open(os.path.join(directory, file_name), "r", encoding="UTF-8") as file:
            # Save file content to dictionary with key of file_name
            file_content[file_name] = file.read()

    return file_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by converting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    stop_words = nltk.corpus.stopwords.words("english")
    return [word.strip(string.punctuation) for word in nltk.word_tokenize(document.lower()) if (word in stop_words or word.strip(string.punctuation) == "") is not True]


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    num_docs = len(documents)

    # Get list of words that appear at least once with no repeats
    unique_words = list(set().union(*documents.values()))

    # Loop all unique words
    for word in unique_words:
        # Calculate idf for unique word
        idfs[word] = math.log(num_docs / sum([1.0 for words in documents.values() if word in words]))

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Sort files based on sum of tf-idf values of words in the query
    sorted_files = sorted(files.keys(), key=lambda file_name: query_tf_idf(query, files[file_name], idfs), reverse=True)

    # Return first n number of elements in the list
    return sorted_files[:n]


def query_tf_idf(query, words, idfs):
    # Get sum of tf-idf of query words in words
    return sum([words.count(query_word) * idfs[query_word] for query_word in query])


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    # Sort sentences based on sum of idf values of words in the query
    sorted_sentences = sorted(sentences.keys(), key=lambda sentence: sentence_idf(query, sentences[sentence], idfs), reverse=True)

    # Return first n number of elements in the list
    return sorted_sentences[:n]


def sentence_idf(query, words, idfs):
    # Get a tuple of the sum of idf of query words in words, and the query term density
    return (sum([idfs[query_word] for query_word in query if query_word in words]), sum([1.0 for word in words if word in query]) / len(words))


if __name__ == "__main__":
    main()
