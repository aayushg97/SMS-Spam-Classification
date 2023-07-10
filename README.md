# SMS Spam Classification 

This project uses pre-trained language models from Hugging Face's Transformers library to classify SMS messages as spam or non-spam.

## Data Collection

<b>Summary :- </b>The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research. This dataset contains 5,574 English messages tagged as either spam ("1") or non-spam ("0").

<b>Classification task :- </b>The text classification task is to take an SMS message as input and determine whether the message is spam ("1") or not ("0"). There are several factors that make the task non-trivial. Spam messages cannot be identified just by looking for some fixed words like 'good', 'bad', 'spam', etc. It is not just the words but the combination of words and the context in which they are used which decides whether the message is spam or not. For example the following message is spam <br><br>
"Sunshine Quiz Wkly Q! Win a top Sony DVD player if u know which country the Algarve is in? Txt ansr to 82277. £1.50 SP:Tyrone"<br><br>
Such messages cannot be identified just by looking for a predefined set of words. These cases require use of complex models that can learn patterns in messages and identify spam. This makes spam detection a non-trivial task.

<b>Statistics :-</b><br>
Labeled examples = 5574<br>
Examples labeled spam = 747<br>
Examples labeled non-spam = 4827<br>
Unique words = 17929<br>

## Model details

This project uses Hugging Face's Transformers (https://github.com/huggingface/transformers), an open-source library that provides general-purpose architectures for natural language understanding and generation with a collection of various pretrained models made by the NLP community. This library allows us to easily use pretrained models like `BERT` and perform experiments on top of them. These models can be used to solve downstream target tasks, such as text classification, question answering, and sequence labeling.

## Hyperparameter tuning

<b>Hyperparameter selection process :- </b>Grid search method was used for hyperparameter tuning. In this method a grid of all possible combinations of hyperparameters was constructed. Then the pre-trained BERT model was trained (fine-tuned) and evaluated for every combination of hyperparameters on the validation set. The combination of hyperparameters that produced the best-performing model on the validation set was then selected as the optimal set of hyperparameters.

Range of hyperparameters considered for grid search is as follows :-

1. Batch size :- {32, 64, 128}
2. Learning rate :- {1e-3, 5e-3, 1e-2, 5e-2}
3. Epsilon :- {1e-7, 1e-8, 1e-9}
4. Epochs :- {3, 5, 10}

<b>Why are chosen hyperparameters better :-</b>

Hyper parameters can affect model performance in many ways :-

1. High learning rate causes the model to diverge and models with very small learning rate fail to reach the optimum in given epochs.

2. Training for too many epochs may make the model overfit the training data.

3. Smaller batch size provides better generalization but if it is too small, the model may see noisy gradients leading to unstable training.

In summary, chosen hyperparameters work better than others because the chosen values are neither too high nor too low and just right (based on the points mentioned above) for obtaining the best accuracy on the validation set.

## Evaluation

| | Accuracy |
| --- | --- |
| Validation set | 0.8725 |
| Test set | 0.8495 |

<b>Discrepancy between test and val accuracy :- </b>There is a 3% gap between validation and test accuracy. This can happen when validation set and test set have different distributions. This also happens when the model overfits the validation set.

## Error analysis

5 incorrect predictions from the model are as follows :-

| Message | Prediction | Label |
| --- | --- | --- |
| You have 1 new voicemail. Please call 08719181513. | 0 | 1 |
| SMS. ac JSco: Energy is high but u may not know where 2channel it. 2day ur leadership skills r strong. Psychic? Reply ANS w/question. End? Reply END JSCO | 0 | 1 |
| Cashbin.co.uk (Get lots of cash this weekend!) www.cashbin.co.uk Dear Welcome to the weekend We have got our biggest and best EVER cash give away!! These.. | 0 | 1 |
| Urgent Please call 09066612661 from landline. £5000 cash or a luxury 4* Canary Islands Holiday await collection. T&Cs SAE award. 20M12AQ. 150ppm. 16+ “ | 0 | 1 |
| Dear Voucher Holder 2 claim this weeks offer at your PC go to http://www.e-tlp.co.uk/expressoffer Ts&Cs apply.2 stop texts txt STOP to 80062. | 0 | 1 |

All 5 examples have label 1 which means that they are spam but the model classifies them as non-spam. This is probably because the dataset is not balanced i.e. number of spam messages in the dataset is significantly less than the number of non-spam messages. An unbalanced distribution is a probable cause of low accuracy.

A possible future step to improve the classifier would be to use a weighted loss function during training. This involves assigning a higher weight to spam messages, which can prevent the model from being biased towards non-spam messages. This way, the penalty of misclassifying a spam message will be higher than misclassifying a non-spam message and the model can efficiently learn to classify spam messages with higher accuracy.

Another way to address this issue is to use ensemble methods to combine predictions of multiple models trained on different subsets of the data. This can improve the model's performance for spam messages by reducing the impact of noise and biases in the individual models.
