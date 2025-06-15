import argparse
import json

from sklearn.metrics import f1_score, precision_score, recall_score

from main import annotation_keys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename", type=str, default="predictions.json",
        help="Filename of the predictions to evaluate"
    )
    args = parser.parse_args()
    with open(args.filename, "r") as f:
        predictions = json.load(f)
    
    all_pred = []
    all_ref = []
    
    for i, value in enumerate(annotation_keys):
        pred = []
        ref = []
        for prediction in predictions:
            pred.append(predictions[prediction]["predicted"][i])
            ref.append(predictions[prediction]["correct_labels"][i])
        all_pred.extend(pred)
        all_ref.extend(ref)

        print(value)

        print("F1 Score:  ", round(f1_score(ref, pred, average="macro"), 2))
        print("Precision: ", round(precision_score(ref, pred, average="macro"), 2))
        print("Recall:    ", round(recall_score(ref, pred, average="macro"), 2))
        print()

    print("ALL")
    print("F1 Score:  ", round(f1_score(all_ref, all_pred, average="macro"), 2))
    print("Precision: ", round(precision_score(all_ref, all_pred, average="macro"), 2))
    print("Recall:    ", round(recall_score(all_ref, all_pred, average="macro"), 2))


if __name__ == "__main__":
    main()
