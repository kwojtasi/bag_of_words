import json
import re
import numpy as np

from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from scipy.stats import mannwhitneyu
from sklearn.svm import SVC
import pandas

SCIPY_METHODS = ['MannWhitney', 'WilcoxonRankSum', 'KolmogorovSmirnov', 'KruskalWallis', 'TTest']
WEKA_METHODS = ['InfoGainAttributeEval', 'CfsSubsetEval', 'ConsistencySubsetEval',
                'GainRatioAttributeEval', 'ChiSquaredAttributeEval']
ENSAMBLE_METHODS =['ExtraTrees','RandomForest','GradientBoosting','AdaBoost']
RFE_METHODS=['RFE_LogisticRegression','RFE_SVM','RFE_NaiveBayes']


class BagOfWords:
    def __init__(self, path_to_file='weighted.json'):
        with open(path_to_file) as json_ifs:
            jsonVal = json.load(json_ifs)
            self.rowlabels = jsonVal["rowlabels"]
            self.collabels = np.asarray(jsonVal["collabels"])
            self.data = np.asarray(jsonVal["arr"])

        self.only_labels = [re.match("(.*)_", label).group(1) for label in self.rowlabels]

    def important_words(self, model, category, num=5, **kwargs):
        word_regex = re.compile('base:(.*)')
        category_labels = [label if label == category else 'inne' for label in self.only_labels]
        dyp_model = model(**kwargs)
        dyp_model.fit(self.data, category_labels)
        importances = dyp_model.feature_importances_
        result = [{word_regex.match(self.collabels[i]).group(1): importances[i]}for i in np.argsort(importances)[::-1][:num]]
        return result

    def rfe_importance(self, category, num=5):
        # TODO
        lg = LogisticRegression()
        category_labels = [label if label == category else 'inne' for label in self.only_labels]
        rfe_lg = RFE(lg, num)
        rfe_lg.fit(self.data, category_labels)
        return self.collabels[rfe_lg.support_]

    def mannwhitneyu_statistic_method(self, category, num=5, **kwargs):
        word_regex = re.compile('base:(.*)')
        category_labels = [1 if label == category else 0 for label in self.only_labels]
        res_list = []

        for i, word in enumerate(self.collabels):
            stat, pvalue = mannwhitneyu(self.data[:, i], category_labels)

            res_list.append((word_regex.match(word).group(1), stat, pvalue))
        res_list.sort(key=lambda x: x[2], reverse=True)
        return res_list[:num]

    @property
    def categories(self):
        return sorted(set(self.only_labels))

    def tsne_plot(self, **tsne_kwargs):
        print(tsne_kwargs)
        plot_params = "".join("({}={})".format(k, v) for k, v in tsne_kwargs.items())
        tsne = TSNE(n_components=2, **tsne_kwargs)
        data_2d = tsne.fit_transform(self.data)
        plt.figure(figsize=(6, 6))
        colors = ('r', 'g', 'b', 'c', 'm')
        for c, label in zip(colors, self.categories):
            indexes = [i for i in range(len(self.only_labels)) if self.only_labels[i] == label]
            plt.scatter(data_2d[indexes, 0], data_2d[indexes, 1], c=c, label=label)
        plt.legend()

        plt.title(plot_params)
        plt.show()

        plt.savefig('/plots/tsne{}'.format(plot_params))

    def word_cloud(self, words, fig_name="word_cloud.png"):
        stopwords = set(STOPWORDS)
        wordcloud = WordCloud(
            background_color='white',
            stopwords=stopwords,
            max_words=20,
            max_font_size=80,
            random_state=42
        ).generate_from_frequencies(words)
        fig = plt.figure(1)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
        fig.savefig(fig_name, dpi=900)


def main():
    bow = BagOfWords(path_to_file='weighted.json')
    cat = 'Katastrofy i Wypadki'
    model = RandomForestClassifier
    word_list = bow.important_words(model, cat, num=20)
    word_dict = dict([(key, word[key]) for word in word_list for key in word])
    bow.word_cloud(word_dict)


def save_to_file():
    bow = BagOfWords(path_to_file='wiki_bow.json')
    model = RandomForestClassifier
    output_filename = 'wiki_result_ada.txt'

    with open(output_filename, 'w') as file:
        file.write('Used model: {}\n'.format(model.__name__))
        for cat in bow.categories:
            file.write("Words for category: {}\n".format(cat))
            file.write(str(bow.important_words(model, cat, num=10)))
            file.write("\n\n")


def different_models():
    bow = BagOfWords(path_to_file='weighted.json')
    models_list = [RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier]
    output_filename = 'different_models_dyplomacja.txt'
    category = 'Dyplomacja'

    with open(output_filename, 'w') as file:
        file.write("Words for category: {}\n".format(category))
        for model in models_list:
            file.write('Used model: {}\n'.format(model.__name__))
            file.write(str(bow.important_words(model, category, num=10)))
            file.write("\n\n")


if __name__ == '__main__':
    bow = BagOfWords(path_to_file='weighted.json')
    result = bow.mannwhitneyu_statistic_method('Ekonomia, Biznes i Finanse')
    print(result)


