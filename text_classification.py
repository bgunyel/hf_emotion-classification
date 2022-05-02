from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import xgboost as xgb

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


class TextClassifier:
    def __init__(self, language_model):
        self.language_model = language_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(self.language_model).to(self.device)

        self.classifier = xgb.XGBClassifier(objective="multi:softprob",
                                            n_estimators=1000,
                                            learning_rate=0.05,
                                            random_state=2020,
                                            tree_method='gpu_hist')

        # Currently UNUSED
        self.classifier_params = {"colsample_bytree": uniform(0.7, 0.3),
                                  "gamma": uniform(0, 0.5),
                                  "learning_rate": uniform(0.03, 0.3),  # default 0.1
                                  "max_depth": randint(2, 6),  # default 3
                                  "n_estimators": randint(100, 150),  # default 100
                                  "subsample": uniform(0.6, 0.4)}

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding=True, truncation=True)

    def extract_hidden_states(self, batch):
        # Place model inputs on the GPU
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in self.tokenizer.model_input_names}

        # Extract last hidden states
        with torch.no_grad():
            last_hidden_state = self.model(**inputs).last_hidden_state

        # Return vector for [CLS] token
        return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

    def train(self, data_train, data_val):
        train_encoded = data_train.map(self.tokenize, batched=True, batch_size=None)
        train_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        train_hidden = train_encoded.map(self.extract_hidden_states, batched=True)

        val_encoded = data_val.map(self.tokenize, batched=True, batch_size=None)
        val_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        val_hidden = val_encoded.map(self.extract_hidden_states, batched=True)

        X_train = np.array(train_hidden["hidden_state"])
        y_train = np.array(train_hidden["label"])

        X_val = np.array(val_hidden["hidden_state"])
        y_val = np.array(val_hidden["label"])

        # random_cv = RandomizedSearchCV(self.classifier, param_distributions=self.classifier_params, random_state=42,
        #                               n_iter=200, cv=3, verbose=1, n_jobs=-1, return_train_score=True)
        # random_cv.fit(X_train, y_train)


        self.classifier.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_val, y_val)])

    def predict(self, dataset):
        dataset_encoded = dataset.map(self.tokenize, batched=True, batch_size=None)
        dataset_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        dataset_hidden = dataset_encoded.map(self.extract_hidden_states, batched=True)

        X_pred = np.array(dataset_hidden["hidden_state"])
        y_pred = self.classifier.predict(X_pred)

        return y_pred
