import json

from flask import Flask
import aps_classifier
clf = aps_classifier.aps_clf()
clf.load("Model")
preproc = aps_classifier.pic_features_extractor()
app = Flask(__name__)

@app.route('/demo/<path:name>')
def demo(name):
    X = preproc.get_features(name)
    label = clf.predict([X])
    return """<h1>Label of picture: {} - {}<h1><img src='{}' alt="sample" width=400px height=auto>""".format(label[0],clf.labels[label[0]],name)

@app.route("/<path:name>")
def test(name):
    X = preproc.get_features(name)
    label = clf.predict([X])[0]
    probs = clf.predict_proba([X])[0]
    output = {
        "label":label,
        "probs":probs,
        "labels":clf.labels
    }
    print(output)
    # output = json.dumps(output)
    return str(output)

if __name__ == '__main__':
    app.run()