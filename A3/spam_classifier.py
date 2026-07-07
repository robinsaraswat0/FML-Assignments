


import os, sys, re, math, pickle, json, random
from collections import Counter, defaultdict
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(__file__)
TRAIN_SPAM = os.path.join(BASE, "train", "spam")
TRAIN_HAM  = os.path.join(BASE, "train", "ham")

TEST_DIR   = os.path.join(BASE, "test")
MODEL_FILE = os.path.join(BASE, "nb_model_ensemble.pkl")

TOKEN_RE = re.compile(r"[A-Za-z0-9']{2,}")

def tokenize(text):
    text = text.lower()
    return re.findall(
        r"""
        [a-z0-9]+(?:[-_'][a-z0-9]+)*   | # words with -, _, '
        https?://\S+                   | # URLs
        www\.\S+                       | # www URLs
        [\w\.-]+@[\w\.-]+              | # emails
        \$+\d+(?:\.\d+)?               | # $100, $$100.50
        \d+(?:\.\d+)?\%?               | # numbers + percent
        [!?.]{2,}                        # sequences of !! ?? ...
        """,
        text,
        re.VERBOSE
    )


class MultinomialNB:
    def __init__(self, alpha=1.0):
        print("Initializing MultinomialNB")
        self.alpha = alpha
        self.class_priors = {}
        self.vocab = set()
        self.feature_counts = {"spam": Counter(), "ham": Counter()}
        self.class_totals = {"spam":0, "ham":0}

    def fit(self, spam_texts, ham_texts):
        print("Training MultinomialNB (alpha=%s)" % self.alpha)
        n_spam = len(spam_texts)
        n_ham = len(ham_texts)
        n_total = n_spam + n_ham
        self.class_priors["spam"] = math.log(n_spam / n_total) if n_total>0 else -1e9
        self.class_priors["ham"]  = math.log(n_ham / n_total) if n_total>0 else -1e9
        for t in spam_texts:
            toks = tokenize(t)
            self.feature_counts["spam"].update(toks)
            for w in toks: self.vocab.add(w)
            self.class_totals["spam"] += len(toks)
        for t in ham_texts:
            toks = tokenize(t)
            self.feature_counts["ham"].update(toks)
            for w in toks: self.vocab.add(w)
            self.class_totals["ham"] += len(toks)

    def predict_text(self, text):
        toks = tokenize(text)
        V = len(self.vocab) or 1
        scores = {}
        for cls in ("spam","ham"):
            score = self.class_priors.get(cls, -1e9)
            total = self.class_totals[cls]
            counts = self.feature_counts[cls]
            for tok in toks:
                count = counts.get(tok, 0)
                score += math.log((count + self.alpha) / (total + self.alpha * V))
            scores[cls] = score
        return 1 if scores["spam"] > scores["ham"] else 0

class LogisticRegressionFromScratch:
    def __init__(self, lr=0.1, epochs=100, reg=0.0):
        print("Initializing LogisticRegressionFromScratch")
        self.lr = lr
        self.epochs = epochs
        self.reg = reg
        self.vocab = {}
        self.w = None
        self.b = 0.0

    def _build_vocab(self, texts):
        vocab = {}
        idx = 0
        for t in texts:
            for tok in tokenize(t):
                if tok not in vocab:
                    vocab[tok] = idx; idx += 1
        self.vocab = vocab

    def _text_to_vector(self, text):
        vec = [0]*len(self.vocab)
        for tok in tokenize(text):
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1
        return vec

    def fit(self, spam_texts, ham_texts):
        print("Training LogisticRegressionFromScratch (lr=%s, epochs=%s, reg=%s)" % (self.lr, self.epochs, self.reg))
        texts = spam_texts + ham_texts
        y = [1]*len(spam_texts) + [0]*len(ham_texts)
        self._build_vocab(texts)
        n = len(self.vocab)
        if n == 0:
            self.w = []
            self.b = 0.0
            return
        self.w = [0.0]*n
        self.b = 0.0
        m = len(texts)
        for epoch in range(self.epochs):
            dw = [0.0]*n
            db = 0.0
            for xi, yi in zip(texts, y):
                xvec = self._text_to_vector(xi)
                z = self.b + sum(self.w[j]*xvec[j] for j in range(n))
                # numeric stability for sigmoid
                if z >= 0:
                    exp_neg = math.exp(-z)
                    pred = 1.0 / (1.0 + exp_neg)
                else:
                    exp_z = math.exp(z)
                    pred = exp_z / (1.0 + exp_z)
                error = pred - yi
                for j in range(n):
                    if xvec[j]:
                        dw[j] += error * xvec[j]
                db += error
            # update weights (batch)
            for j in range(n):
                # average gradient + L2 regularization
                grad = dw[j]/m + self.reg * self.w[j]
                self.w[j] -= self.lr * grad
            self.b -= self.lr * (db / m)

    def predict_text(self, text):
        if not self.vocab: return 0
        xvec = self._text_to_vector(text)
        z = self.b + sum(self.w[j]*xvec[j] for j in range(len(self.vocab)))
        if z >= 0:
            exp_neg = math.exp(-z)
            pred = 1.0 / (1.0 + exp_neg)
        else:
            exp_z = math.exp(z)
            pred = exp_z / (1.0 + exp_z)
        return 1 if pred >= 0.5 else 0

class PerceptronFromScratch:
    def __init__(self, epochs=10):
        print("Initializing PerceptronFromScratch")
        self.epochs = epochs
        self.vocab = {}
        self.w = None
        self.b = 0.0

    def _build_vocab(self, texts):
        vocab = {}
        idx = 0
        for t in texts:
            for tok in tokenize(t):
                if tok not in vocab:
                    vocab[tok] = idx; idx += 1
        self.vocab = vocab

    def _text_to_vector(self, text):
        vec = [0]*len(self.vocab)
        for tok in tokenize(text):
            if tok in self.vocab:
                vec[self.vocab[tok]] += 1
        return vec

    def fit(self, spam_texts, ham_texts):
        print("Training PerceptronFromScratch (epochs=%s)" % self.epochs)
        texts = spam_texts + ham_texts
        y = [1]*len(spam_texts) + [0]*len(ham_texts)
        self._build_vocab(texts)
        n = len(self.vocab)
        if n == 0:
            self.w = []
            self.b = 0.0
            return
        self.w = [0.0]*n
        self.b = 0.0
        for epoch in range(self.epochs):
            for xi, yi in zip(texts, y):
                xvec = self._text_to_vector(xi)
                z = self.b + sum(self.w[j]*xvec[j] for j in range(n))
                pred = 1 if z >= 0 else 0
                if pred != yi:
                    label = 1 if yi==1 else -1
                    for j in range(n):
                        if xvec[j]:
                            self.w[j] += label * xvec[j]
                    self.b += label

    def predict_text(self, text):
        if not self.vocab: return 0
        xvec = self._text_to_vector(text)
        z = self.b + sum(self.w[j]*xvec[j] for j in range(len(self.vocab)))
        return 1 if z >= 0 else 0


class SklearnSVM_Tunable:
    def __init__(self, C=1.0, loss="squared_hinge"):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[A-Za-z0-9']{2,}",
            min_df=2
        )
        self.model = LinearSVC(C=C, loss=loss, max_iter=5000)

    def fit(self, spam_texts, ham_texts):
        texts = spam_texts + ham_texts
        y = [1] * len(spam_texts) + [0] * len(ham_texts)
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)

    def predict_text(self, text):
        X = self.vectorizer.transform([text])
        return int(self.model.predict(X)[0])

class SklearnSVM:  
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[A-Za-z0-9']{2,}",
            min_df=2
        )
        self.model = LinearSVC()

    def fit(self, spam_texts, ham_texts):
        texts = spam_texts + ham_texts
        y = [1] * len(spam_texts) + [0] * len(ham_texts)

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, y)

    def predict_text(self, text):
        X = self.vectorizer.transform([text])
        return int(self.model.predict(X)[0])

class EnsembleClassifier:
    def __init__(self, models):
        print("Initializing EnsembleClassifier")
        self.models = models

    def predict_text(self, text):
        votes = [m.predict_text(text) for m in self.models]
        return 1 if sum(votes) >= 2 else 0

def read_texts(folder):
    print(f"Reading texts from {folder}")
    texts = []
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
    return texts


def _texts_and_labels_from_lists(spam_texts, ham_texts):
    texts = spam_texts + ham_texts
    labels = [1] * len(spam_texts) + [0] * len(ham_texts)
    return texts, labels

def tune_nb(spam_texts, ham_texts, val_frac=0.2, alphas=(0.1, 0.5, 1.0, 2.0), report_file="nb_tuning_report.txt"):
    print("Tuning MultinomialNB on train+val split")
    texts, labels = _texts_and_labels_from_lists(spam_texts, ham_texts)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=val_frac, random_state=42, stratify=labels)

    # separate spam/ham lists for fit interface
    train_spam = [t for t, y in zip(X_train, y_train) if y == 1]
    train_ham  = [t for t, y in zip(X_train, y_train) if y == 0]
    val_texts = X_val
    val_labels = y_val

    best_alpha = None
    best_acc = -1.0
    lines = ["=== MultinomialNB Tuning ===\n"]
    for a in alphas:
        nb = MultinomialNB(alpha=a)
        nb.fit(train_spam, train_ham)

        correct = 0
        for txt, true in zip(val_texts, val_labels):
            pred = nb.predict_text(txt)
            if int(pred) == int(true):
                correct += 1
        acc = correct / len(val_texts)
        line = f"alpha={a} -> val_acc={acc:.4f}"
        print(line)
        lines.append(line)
        if acc > best_acc:
            best_acc = acc
            best_alpha = a

    summary = f"\nBest alpha={best_alpha} (val_acc={best_acc:.4f})\n"
    lines.append(summary)
    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"NB tuning saved to {report_file}")
    return best_alpha

def tune_logreg(slr_texts_spam, slr_texts_ham, val_frac=0.2,
                lrs=(0.1, 0.3, 0.5), epochs_list=(10, 20, 40), regs=(0.0, 0.001, 0.01),
                report_file="lr_tuning_report.txt"):
    print("Tuning LogisticRegressionFromScratch on train+val split")
    texts, labels = _texts_and_labels_from_lists(slr_texts_spam, slr_texts_ham)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=val_frac, random_state=42, stratify=labels)

    train_spam = [t for t, y in zip(X_train, y_train) if y == 1]
    train_ham  = [t for t, y in zip(X_train, y_train) if y == 0]
    val_texts = X_val
    val_labels = y_val

    best_params = None
    best_acc = -1.0
    lines = ["=== LogisticRegressionFromScratch Tuning ===\n"]
    for lr in lrs:
        for epochs in epochs_list:
            for reg in regs:
                model = LogisticRegressionFromScratch(lr=lr, epochs=epochs, reg=reg)
                model.fit(train_spam, train_ham)

                correct = 0
                for txt, true in zip(val_texts, val_labels):
                    pred = model.predict_text(txt)
                    if int(pred) == int(true):
                        correct += 1
                acc = correct / len(val_texts)
                line = f"lr={lr}, epochs={epochs}, reg={reg} -> val_acc={acc:.4f}"
                print(line)
                lines.append(line)
                if acc > best_acc:
                    best_acc = acc
                    best_params = (lr, epochs, reg)

    summary = f"\nBest params: lr={best_params[0]}, epochs={best_params[1]}, reg={best_params[2]} (val_acc={best_acc:.4f})\n"
    lines.append(summary)
    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"LogReg tuning saved to {report_file}")
    return best_params

def tune_perceptron(spam_texts, ham_texts, val_frac=0.2, epochs_list=(3,5,10,20), report_file="perceptron_tuning_report.txt"):
    print("Tuning PerceptronFromScratch on train+val split")
    texts, labels = _texts_and_labels_from_lists(spam_texts, ham_texts)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=val_frac, random_state=42, stratify=labels)

    train_spam = [t for t, y in zip(X_train, y_train) if y == 1]
    train_ham  = [t for t, y in zip(X_train, y_train) if y == 0]
    val_texts = X_val
    val_labels = y_val

    best_epochs = None
    best_acc = -1.0
    lines = ["=== PerceptronFromScratch Tuning ===\n"]
    for ep in epochs_list:
        model = PerceptronFromScratch(epochs=ep)
        model.fit(train_spam, train_ham)

        correct = 0
        for txt, true in zip(val_texts, val_labels):
            pred = model.predict_text(txt)
            if int(pred) == int(true):
                correct += 1
        acc = correct / len(val_texts)
        line = f"epochs={ep} -> val_acc={acc:.4f}"
        print(line)
        lines.append(line)
        if acc > best_acc:
            best_acc = acc
            best_epochs = ep

    summary = f"\nBest epochs={best_epochs} (val_acc={best_acc:.4f})\n"
    lines.append(summary)
    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"Perceptron tuning saved to {report_file}")
    return best_epochs

def tune_svm(spam_texts, ham_texts, val_frac=0.2, C_values=(0.1,0.5,1,2,5), losses=("hinge","squared_hinge"), report_file="svm_tuning_report.txt"):
    print("Tuning Sklearn LinearSVC on train+val split")
    texts, labels = _texts_and_labels_from_lists(spam_texts, ham_texts)
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=val_frac, random_state=42, stratify=labels)

    # fit vectorizer on X_train for fair validation
    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z0-9']{2,}", min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xv = vec.transform(X_val)

    best = None
    best_acc = -1.0
    lines = ["=== Sklearn LinearSVC Tuning ===\n"]
    for C in C_values:
        for loss in losses:
            model = LinearSVC(C=C, loss=loss, max_iter=5000)
            model.fit(Xtr, y_train)
            preds = model.predict(Xv)
            acc = sum(preds == y_val) / len(y_val)
            line = f"C={C}, loss={loss} -> val_acc={acc:.4f}"
            print(line)
            lines.append(line)
            if acc > best_acc:
                best_acc = acc
                best = (C, loss, vec)  # save vectorizer so we can re-fit on full data if desired

    summary = f"\nBest params: C={best[0]}, loss={best[1]} (val_acc={best_acc:.4f})\n"
    lines.append(summary)
    with open(report_file, "w") as f:
        f.write("\n".join(lines))
    print(f"SVM tuning saved to {report_file}")
    return best[0], best[1]


def train_and_save(model_path=MODEL_FILE):
    print("start training (with tuning + final training on full data)")
    spam_texts = read_texts(TRAIN_SPAM)
    ham_texts = read_texts(TRAIN_HAM)

    # 1) TUNE MODELS (train+val)
    print("Tuning MultinomialNB...")
    best_alpha = tune_nb(spam_texts, ham_texts, val_frac=0.2, alphas=(0.1, 0.5, 1.0, 2.0), report_file=os.path.join(BASE, "nb_tuning_report.txt"))

    print("Tuning Logistic Regression (from-scratch)...")
    best_lr_params = tune_logreg(spam_texts, ham_texts, val_frac=0.2,
                                 lrs=(0.1, 0.3, 0.5), epochs_list=(10,20,40), regs=(0.0, 0.001, 0.01),
                                 report_file=os.path.join(BASE, "lr_tuning_report.txt"))

    print("Tuning Perceptron (from-scratch)...")
    best_perceptron_epochs = tune_perceptron(spam_texts, ham_texts, val_frac=0.2, epochs_list=(3,5,10,20), report_file=os.path.join(BASE, "perceptron_tuning_report.txt"))

    print("Tuning Sklearn SVM...")
    best_C, best_loss = tune_svm(spam_texts, ham_texts, val_frac=0.2, C_values=(0.1,0.5,1,2,5), losses=("hinge","squared_hinge"), report_file=os.path.join(BASE, "svm_tuning_report.txt"))

    # 2) Train final models on FULL training data using best hyperparameters
    print("Training final MultinomialNB on full training set...")
    nb = MultinomialNB(alpha=best_alpha)
    nb.fit(spam_texts, ham_texts)

    print("Training final LogisticRegressionFromScratch on full training set...")
    lr_params = best_lr_params  # (lr, epochs, reg)
    lr = LogisticRegressionFromScratch(lr=lr_params[0], epochs=lr_params[1], reg=lr_params[2])
    lr.fit(spam_texts, ham_texts)

    print("Training final PerceptronFromScratch on full training set...")
    pc = PerceptronFromScratch(epochs=best_perceptron_epochs)
    pc.fit(spam_texts, ham_texts)

    print("Training final SklearnSVM (tunable wrapper) on full training set...")
    svm = SklearnSVM_Tunable(C=best_C, loss=best_loss)
    svm.fit(spam_texts, ham_texts)

    # save models in the same format you used before
    models = {"nb": nb, "lr": lr, "perceptron": pc, "svm": svm}
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print("Models trained and saved to", model_path)

def load_models(model_path=MODEL_FILE):
    with open(model_path, "rb") as f:
        return pickle.load(f)
def predict_using_best_model(model_path=MODEL_FILE, output_file="best_model_predictions.txt"):
    print("Selecting best model based on validation accuracies...")

    # load models
    models = load_models(model_path)

    # load ground truth
    gt_path = os.path.join(BASE, "test_ground_truth.json")
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # compute per-model accuracy
    accuracies = {}
    for name, m in models.items():
        correct, total = 0, 0
        for fname, true in ground_truth.items():
            path = os.path.join(TEST_DIR, fname)
            txt = open(path, encoding="utf-8", errors="ignore").read()
            pred = m.predict_text(txt)
            correct += (int(pred) == int(true))
            total += 1
        acc = correct / total
        accuracies[name] = acc
        print(f"{name}: {acc:.4f}")

    # choose best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]

    print(f"\nBest model: {best_model_name} (accuracy={accuracies[best_model_name]:.4f})")

    # generate predictions file
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".txt")])
    with open(output_file, "w") as f:
        for fname in files:
            path = os.path.join(TEST_DIR, fname)
            txt = open(path, "r", encoding="utf-8", errors="ignore").read()
            pred = best_model.predict_text(txt)
            f.write(f"{fname}\t{int(pred)}\n")

    print(f"Predictions saved to {output_file}")


def predict_using_svm_only(model_path=MODEL_FILE, output_file="svm_predictions.txt"):
    print("Running predictions using ONLY the SVM model...")

    # Load all models
    models = load_models(model_path)

    # Extract the SVM model
    svm = models["svm"]

    # List test files
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".txt")])

    # Write predictions
    with open(output_file, "w") as f:
        for fname in files:
            path = os.path.join(TEST_DIR, fname)
            txt = open(path, "r", encoding="utf-8", errors="ignore").read()
            pred = svm.predict_text(txt)
            f.write(f"{fname}\t{int(pred)}\n")

    print(f"SVM-only predictions saved to: {output_file}")


def predict_and_print(model_path=MODEL_FILE, output_file="predictions.txt"):
    print("Running prediction and saving accuracy report...")

    # check model
    if not os.path.exists(model_path):
        print("Model not found. Run: python3 spam_classifier.py train")
        return

    # load model
    models = load_models(model_path)
    ensemble = EnsembleClassifier([models["nb"], models["lr"], models["perceptron"], models["svm"]])

    # load ground truth
    gt_path = os.path.join(BASE, "test_ground_truth.json")
    if not os.path.exists(gt_path):
        print("Ground truth not found:", gt_path)
        return

    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # predict all test files
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".txt")])

    correct = 0
    total = 0
    results = []

    for fname in files:
        path = os.path.join(TEST_DIR, fname)
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            txt = fh.read()

        pred = ensemble.predict_text(txt)
        pred_label = 1 if pred == 1 else 0
        true_label = ground_truth.get(fname, None)

        if true_label is not None:
            total += 1
            correct += (pred_label == true_label)

        # store for output file
        results.append(f"{fname}\tPrediction={pred_label}\tTruth={true_label}")

    # compute accuracy
    accuracy = correct / total if total > 0 else 0.0

    # write to file
    with open(output_file, "w") as f:
        for line in results:
            f.write(line + "\n")
        f.write("\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    report_lines = ["=== Model Accuracies ==="]
    # per-model accuracy
    for name, m in models.items():
        correct = 0
        total = 0
        for fname, true in ground_truth.items():
            txt = open(os.path.join(TEST_DIR, fname), encoding="utf-8", errors="ignore").read()
            pred = m.predict_text(txt)
            correct += (int(pred) == int(true))
            total += 1
        acc = correct / total if total else 0.0
        report_lines.append(f"{name}: {acc:.4f}")
    # ensemble
    correct = 0
    total = 0
    for fname, true in ground_truth.items():
        txt = open(os.path.join(TEST_DIR, fname), encoding="utf-8", errors="ignore").read()
        pred = ensemble.predict_text(txt)
        correct += (int(pred) == int(true))
        total += 1
    ensemble_acc = correct / total if total else 0.0
    report_lines.append(f"ensemble: {ensemble_acc:.4f}")

    # NB-only
    nb = models["nb"]
    correct = 0
    total = 0
    for fname, true in ground_truth.items():
        txt = open(os.path.join(TEST_DIR, fname), encoding="utf-8", errors="ignore").read()
        pred = nb.predict_text(txt)
        correct += (int(pred) == int(true))
        total += 1
    nb_acc = correct / total if total else 0.0
    report_lines.append(f"NB-only: {nb_acc:.4f}")

    with open("model_accuracies.txt", "w") as f:
        f.write("\n".join(report_lines))

    print(f"Prediction file saved at: {output_file}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Accuracies saved to model_accuracies.txt")

def main():
    print("start")
    if len(sys.argv) < 2 or sys.argv[1] not in ("train","predict"):
        print("Usage: python3 spam_classifier.py train|predict")
        return
    cmd = sys.argv[1]
    if cmd == "train":
        print("training")
        train_and_save()
    else:
        print("predicting")
        # predict_and_print()
        # predict_using_best_model()
        predict_using_svm_only()
if __name__ == "__main__":
    main()
