from nltk import word_tokenize
from nltk.stem import PorterStemmer
import joblib


class Text_model():
    def __init__(self, text_):
        self.text = text_

    def prepare_stop_words(self):
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
        stop_words += " was wasn wasnt we were weren werent weve what when where which while who whom why will with won"
        stop_words += " wont would wouldn wouldnt www y yet you youd youll your youre yours yourself yourselves youve yuove"
        stop_words = list(set(stop_words.split(" ")))
        if "" in stop_words:
            stop_words.remove("")
        return stop_words, stop_sep

    def str_prepare(self):
        stop_words, stop_sep = self.prepare_stop_words()
        x = self.text
        x = x.lower().replace("_", " ").replace(" - ", " ").replace('"', ' ').replace(" – ", " "). \
            replace("“", " ").replace("”", " ")
        x = x.expandtabs(1).replace("\n", " ")
        for i in stop_sep:
            x = x.replace(i, " ")
        for i in stop_words:
            x = x.replace(" " + i + " ", " ")
        x = x.replace("trade in", "trade_in")
        x = ' '.join(x.split())
        self.text = x

    def str_lemmatize(self):
        self.str_prepare()
        mystem = PorterStemmer()
        self.text = " ".join([mystem.stem(t) for t in word_tokenize(self.text)])
        return self.text


class Classifier(object):
    def __init__(self):
        self.mlb = joblib.load("mlb.pkl")
        self.vect = joblib.load("tfidf.pkl")
        self.clf = joblib.load("OneVsRest_logr.pkl")

    def predict_genres(self, text_):
        I_MAX = 17
        tm = Text_model(text_)
        vector_for_pred = self.vect.transform([tm.str_lemmatize()])

        text_pred = self.clf.predict_proba(vector_for_pred)
        text_pred = (text_pred.T / text_pred.sum(axis=1)).T

        for i in range(text_pred.shape[0]):
            text_pred[i, text_pred[i, :].argmax()] = 1
        y_text_pred = (text_pred > I_MAX / 100).astype("uint8")
        return list(self.mlb.inverse_transform(y_text_pred)[0])
