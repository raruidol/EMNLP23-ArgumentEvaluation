from tools import argumentation as arg
import pandas as pd
import os
import random
import math
from tools import gcn
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

random.seed(8)
# 82% - 18% debates
# 95% - 5% extensions
N_TRAIN = 23
N_TEST = 5
KFOLD = 0

if __name__ == '__main__':
    networkx_dataset = {}
    graphbl_dataset = {}
    f_samples = 0
    a_samples = 0
    n_samples = 0
    p_samples = 0
    eval = pd.read_csv('data/VivesEval/VivesDebate_eval.csv')
    for _, _, files in os.walk('data/VivesDebate/'):
        for file in sorted(files):
            debate_samples = []
            # print(file)
            debate = pd.read_csv('data/VivesDebate/' +file)

            eval_df = eval.loc[eval['DEBATE'] == file.split('.')[0]]
            try:
                winner = eval.iloc[eval_df['SCORE'].idxmax(), :]

                if winner['STANCE'] == 'Favour':
                    label = 0
                else:
                    label = 1
            except:
                label = 1

            # Generate a graph from the annotated debate
            g = arg.debate_to_graph(debate)
            # Generate an Abstract Argumentation Framework from the graph
            af = arg.graph_to_af(g)
            # Calculate acceptable extensions
            n_ext = arg.naive_semantics(af)
            p_ext = arg.preferred_semantics(af, len(n_ext[0]))

            for ext in n_ext:
                # First, create edges between conflicting args
                sample = arg.extension_to_sample_graph(ext, af)
                # Second, append sample to the dataset
                debate_samples.append([sample, label])
                if label == 0:
                    f_samples += 1
                elif label == 1:
                    a_samples += 1

            networkx_dataset[file] = debate_samples
            graphbl_dataset[file] = [[g, label]]

            # print("Naive: ", len(n_ext), n_ext)
            # print("Preferred: ", len(p_ext), p_ext)

            # print('---------------------------------')
            # print('---------------------------------')

    final_sample = networkx_dataset.pop('Debate7.csv')
    graphbl_dataset.pop('Debate7.csv')

    nx_train_dataset = []
    nx_test_dataset = []

    g_train_dataset = []
    g_test_dataset = []

    # Single evaluation
    if KFOLD == 0:
        td = 0
        for key in networkx_dataset:
            if td < N_TRAIN:
                for sample in networkx_dataset[key]:
                    nx_train_dataset.append(sample)
                td += 1

            else:
                for sample in networkx_dataset[key]:
                    nx_test_dataset.append(sample)

        td = 0
        for key in graphbl_dataset:
            if td < N_TEST:
                for sample in graphbl_dataset[key]:
                    g_test_dataset.append(sample)
                td += 1

            else:
                for sample in graphbl_dataset[key]:
                    g_train_dataset.append(sample)

        print("Dataset number of samples:", len(nx_train_dataset)+len(nx_test_dataset))
        print("0 samples:", f_samples, "1 samples:", a_samples)
        os = [o for o in nx_train_dataset if o[1] == 0]
        Is = [i for i in nx_train_dataset if i[1] == 1]
        print("Train 0 samples:", len(os))
        print("Train 1 samples:", len(Is))
        os = [o for o in nx_test_dataset if o[1] == 0]
        Is = [i for i in nx_test_dataset if i[1] == 1]
        print("Test 0 samples:", len(os))
        print("Test 1 samples:", len(Is))

        jraph_train_data = gcn.convert_networkx_dataset_to_jraph(nx_train_dataset)
        jraph_test_data = gcn.convert_networkx_dataset_to_jraph(nx_test_dataset)
        jraph_final_sample = gcn.convert_networkx_dataset_to_jraph(final_sample)

        jraph_g_train_data = gcn.convert_graph_dataset_to_jraph(g_train_dataset)
        jraph_g_test_data = gcn.convert_graph_dataset_to_jraph(g_test_dataset)

        print('Number of Training samples for GCN: ', len(jraph_train_data))
        print('Number of Test samples for GCN: ', len(jraph_test_data))

        random.shuffle(jraph_train_data)
        random.shuffle(jraph_test_data)
        random.shuffle(jraph_g_train_data)
        random.shuffle(jraph_g_test_data)

        params = gcn.train(jraph_train_data, num_train_steps=2500)
        print('------------------------------------')
        gcn.evaluate(jraph_test_data, params)
        res1 = gcn.preds(jraph_test_data, params)
        y_true = []
        y_pred = []
        for pred in res1:
            y_true.append(pred[1][0])
            y_pred.append(np.argmax(pred[0]))
        print("Test predictions:", res1)
        print(y_true)
        print(y_pred)
        print('Test Accuracy score:', accuracy_score(y_true, y_pred))
        print('Test F1 score:', f1_score(y_true, y_pred, average='macro'))

        print('------------------------------------')
        print(final_sample)
        res = gcn.preds(jraph_final_sample, params)
        print("Debate Sample predictions:", res)
        gcn.evaluate(jraph_final_sample, params)

        print('------------------------------------')
        params1 = gcn.train(jraph_g_train_data, num_train_steps=100)
        # Language Model Baseline
        gcn.evaluate(jraph_g_test_data, params1)
        res2 = gcn.preds(jraph_g_test_data, params1)
        y_true_lm = []
        y_pred_lm = []
        for pred in res2:
            y_true_lm.append(pred[1][0])
            y_pred_lm.append(np.argmax(pred[0]))
        print("LMBaseline predictions:", res2)
        print(y_true_lm)
        print(y_pred_lm)
        print('LMBaseline Accuracy score:', accuracy_score(y_true_lm, y_pred_lm))
        print('LMBaseline F1 score:', f1_score(y_true_lm, y_pred_lm, average='macro'))

        print('------------------------------------')
        # Random Baseline
        rnd_score = 0
        y_true_rb = []
        y_pred_rb = []
        for spl in nx_test_dataset:
            label = random.randint(0, 1)
            y_true_rb.append(spl[1])
            y_pred_rb.append(label)
            if label == spl[1]:
                rnd_score += 1

        print('Random baseline accuracy:', rnd_score/len(nx_test_dataset))
        print('Random baseline F1 score:', f1_score(y_true_rb, y_pred_rb, average='macro'))

        print('------------------------------------')
        # Argumentation Theory baseline
        atb_score = 0
        y_true_atb = []
        y_pred_atb = []
        for spl in nx_test_dataset:
            nodes_favour = (
                node
                for node, data
                in spl[0].nodes(data=True)
                if data.get("team") == 'FAVOUR'
            )
            nf = len(list(spl[0].subgraph(nodes_favour)))

            nodes_against = (
                node
                for node, data
                in spl[0].nodes(data=True)
                if data.get("team") == 'AGAINST'
            )
            na = len(list(spl[0].subgraph(nodes_against)))

            if nf > na:
                label = 0
            else:
                label = 1

            y_true_atb.append(spl[1])
            y_pred_atb.append(label)

            if label == spl[1]:
                atb_score += 1

        print('Argumentation Theory baseline accuracy:', atb_score / len(nx_test_dataset))
        print('Random baseline F1 score:', f1_score(y_true_atb, y_pred_atb, average='macro'))

    # KFold evaluation
    else:
        samplexfold = math.ceil(len(networkx_dataset) / KFOLD)
        folded = [[] for _ in range(KFOLD)]
        f = 0
        for key in networkx_dataset:
            folded[f].append(networkx_dataset[key])
            if len(folded[f]) == samplexfold:
                f += 1
        global_acc = 0
        for f in range(KFOLD):
            nx_test_dataset = []
            nx_train_dataset = []
            test = folded.pop(f)
            for s in test:
                for sample in s:
                    nx_test_dataset.append(sample)
            for fold in folded:
                for s in fold:
                    for sample in s:
                        nx_train_dataset.append(sample)

            folded.insert(f, test)

            print(f+1, "FOLD")
            print('Train', len(nx_train_dataset))
            print('Test', len(nx_test_dataset))

            jraph_train_data = gcn.convert_networkx_dataset_to_jraph(nx_train_dataset)
            jraph_test_data = gcn.convert_networkx_dataset_to_jraph(nx_test_dataset)

            random.shuffle(jraph_train_data)
            random.shuffle(jraph_test_data)

            params = gcn.train(jraph_train_data, num_train_steps=10000)
            loss, acc = gcn.evaluate(jraph_test_data, params)
            global_acc += acc

        print(KFOLD, "Fold Evaluation:", global_acc/KFOLD, "accuracy.")
