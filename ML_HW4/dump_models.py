import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import joblib
import warnings
warnings.simplefilter('ignore')


def init_data():
    data = pd.read_csv('train.zip')
    data.drop(columns = "movie", inplace = True)
    for chr in ["[", "]", "," ,"u'","'"]:
        data.genres = data.genres.str.replace(chr, "")
    return data


def prepare_stop_words():
    stop_sep = ", . ' : ; ! ? № % * ( ) [ ] { | } # $ ^ & - + < = > ` ~ 1 2 3 4 5 6 7 8 9 0 | @ · \' - `"
    stop_sep += " · • — ❗️ ✪ \\ / 😁 😊 😉 ∙ ✔ ► ₽ ″ « » … ✅ ☑️ 🤦 ● 🔰 ° 📌 📢 ☎ ▼ ➥ ☛ 。 🔝 ⬇️ ▶"
    stop_sep = stop_sep.split(" ")
    stop_words = "<BR> a able about above across after again against ain all almost also am among an and another any"
    stop_words += " are aren arent as at be because been before being below between both br but by can cannot could"
    stop_words += " couldn couldnt d deardid didn didnt do does doesn doesnt doing don down during each either else"
    stop_words += " ever every few find for from further get going got had hadn hadnt has hasn hasnt have haven havent"
    stop_words += " having he her here hers herself hes him himself his how however https i if im in into is isn isnt"
    stop_words += " it its itself ive just least let like likely ll m ma man may me might mightn mightnt more most must"
    stop_words += " mustn mustnt my myself needn neednt neither no nor not now o of off often on once one only or other"
    stop_words += " our ours ourselves out over own rather re s said same say says shan shant she shes should shouldn"
    stop_words += " shouldnt shouldve since so some such t than that thatll the their theirs them themselves then there"
    stop_words += " these they theyre things this those through tis to too twas two under until up us ve very wants"
    stop_words += " was wasn wasnt we were weren werent weve what when where which while who whom why will with"
    stop_words += " won wont would wouldn wouldnt www y yet you youd youll your youre yours yourself yourselves youve yuove"
    stop_words = list(set(stop_words.split(" ")))
    if "" in stop_words:
        stop_words.remove("")
    return stop_words, stop_sep


def text_check(x, stop_slovo, stop_slovo_sep):
    x = x.lower().replace("_", " ").replace(" - ", " ").replace('"', ' ').replace(" – ", " ").\
        replace("“", " ").replace("”", " ")
    x = x.expandtabs(1).replace("\n", " ")
    for i in stop_slovo_sep:
        x = x.replace(i, " ")
    for i in stop_slovo:
        x = x.replace(" " + i + " ", " ")
    x = x.replace("trade in", "trade_in")
    x = ' '.join(x.split())
    return x


def text_prepare(data, col_name='dialogue'):
    data[col_name].fillna("", inplace=True)
    data[col_name] = data[col_name].astype(str)
    stop_words, stop_sep = prepare_stop_words()
    data[col_name] = data[col_name].apply(lambda x: text_check(x, stop_words, stop_sep))


def text_lemmatize(data, col_name='dialogue'):
    mystem = PorterStemmer()
    data[col_name] = data[col_name].apply(lambda x: " ".join([mystem.stem(t) for t in word_tokenize(x)]))


def calc_model_and_dump():
    # calc model
    I_MAX = 17
    data = init_data()
    text_prepare(data, col_name = 'dialogue')
    text_lemmatize(data, col_name = 'dialogue')
    mlb = MultiLabelBinarizer()
    mlb.fit(data.genres.str.split(" "))
    target = mlb.transform(data.genres.str.split(" "))
    print(list(mlb.classes_), len(mlb.classes_))

    train_data = data['dialogue']
    vect = TfidfVectorizer(min_df = 5, sublinear_tf=True, ngram_range = (1,1))
    vect.fit(train_data)
    xtrain_tfidf = vect.transform(train_data)
    print(xtrain_tfidf.shape)

    lr = LogisticRegression(random_state = 999, n_jobs = -1, penalty = "l2", C = 1, max_iter = 500)
    clf = OneVsRestClassifier(lr)
    clf.fit(xtrain_tfidf, target)

    X_tfidf_pred = vect.transform(train_data)
    test_pred = clf.predict_proba(X_tfidf_pred)
    test_pred = (test_pred.T/test_pred.sum(axis = 1)).T

    for i in range(test_pred.shape[0]):
        test_pred[i, test_pred[i, :].argmax()] = 1
    y_pred = (test_pred > I_MAX/100).astype("uint8")
    result = mlb.inverse_transform(y_pred)
    data['result_logr'] = list(map(lambda x: " ".join(x), result))

    # save model
    with open('mlb.pkl', 'wb') as output_file:
        joblib.dump(mlb, output_file)

    with open('tfidf.pkl', 'wb') as output_file:
        joblib.dump(vect, output_file)

    with open('OneVsRest_logr.pkl', 'wb') as output_file:
        joblib.dump(clf, output_file)

calc_model_and_dump()
