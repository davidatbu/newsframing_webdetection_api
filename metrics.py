import logging
import typing as T

import numpy as np  # type: ignore
import sklearn.metrics as skmetrics  # type: ignore

from processors import LABELS


def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes the average of accuracy computed for each label
    """
    return np.equal(y_true, y_pred).mean(axis=0).mean()


def exact_match_single(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes subset accuracy among those who have only one label
    """
    single_labeled_indices = y_true.sum(axis=1) == 1
    cleaned_y_true = y_true[single_labeled_indices, :]
    cleaned_y_pred = y_pred[single_labeled_indices, :]

    return skmetrics.accuracy_score(cleaned_y_true, cleaned_y_pred)


# TODO: Change terrible naming of y_logits to y_probs
def tweetclf_compute_metrics(
    task_name: str, y_pred: np.ndarray, y_true: np.ndarray
) -> T.Dict[str, float]:
    """
    Computes a dicitonary of metrics once the model is done using the scoring model.
    """

    if y_pred.ndim != 1 and y_pred.ndim == y_true.ndim:
        raise Exception(
            "Inputs must be flat arrays. No onehot support, "
            "if that's what you're looking for."
        )

    if not len(y_pred) == len(y_true):
        raise Exception(
            "Umm, {} true samples, but {} predicted samples".format(
                len(y_true), len(y_pred)
            )
        )

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    # Exclude examples whose labels don't occur in gold or predict
    set_labels = set(range(len(LABELS)))  # The predictions are indices/output of argmax
    labels_not_in_gold = set_labels - set(y_true)
    labels_not_in_pred = set_labels - set(y_pred)
    labels_to_exclude = labels_not_in_gold.union(labels_not_in_pred)
    labels_to_include = list(set_labels.difference(labels_to_exclude))
    labels_to_include.sort()

    logging.info("%s labels excluded because not in gold", labels_not_in_gold)
    logging.info("%s labels excluded because not in pred", labels_not_in_pred)
    # Whether to compute F1 and AUC, which can be done on cleaned_y_true and
    # cleaned_y_pred below
    do_cleaned = True
    if len(labels_to_include) < len(LABELS) / 2:
        do_cleaned = False
        logging.warning(
            "No F1 and AUC scores because the only labels found both in gold and pred were %s",
            labels_to_include,
        )

    cleaned_y_true = y_true[np.isin(y_true, labels_to_include)]
    cleaned_y_pred = y_pred[np.isin(y_true, labels_to_include)]

    logging.debug("Cleaned y_pred: {}".format(cleaned_y_pred))
    logging.debug("Cleaned y_true: {}".format(cleaned_y_true))

    metrics_on_cleaned_only = [
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "auc",
    ]

    metrics_on_preds = [
        ("f1_macro", lambda t, p: skmetrics.f1_score(t, p, average="macro")),
        ("f1_micro", lambda t, p: skmetrics.f1_score(t, p, average="micro")),
        ("f1_weighted", lambda t, p: skmetrics.f1_score(t, p, average="weighted")),
        ("accuracy", lambda t, p: skmetrics.accuracy_score(t, p)),
    ]

    results = {}

    for name, func in metrics_on_preds:
        if name in metrics_on_cleaned_only:
            if do_cleaned:
                results[name] = func(cleaned_y_true, cleaned_y_pred)
        else:
            results[name] = func(y_true, y_pred)

    return results


def average_metrics_across_folds(
    metrics_per_fold: T.Dict[T.Any, T.Dict[str, float]],
    average_func: T.Callable[[T.Iterable[float]], float] = np.mean,
) -> T.Dict[str, float]:
    assert len(metrics_per_fold) > 0

    averaged_metrics = {}
    metric_names = list(metrics_per_fold.values())[0].keys()
    for metric_name in metric_names:
        averaged_metrics[metric_name] = average_func(
            [fold_metrics[metric_name] for fold_metrics in metrics_per_fold.values()]
        )
    return averaged_metrics


def main():

    logging.root.setLevel(logging.DEBUG)

    y_pred = np.array([1, 2, 3, 3, 1])
    y_true = np.array([1, 2, 3, 3, 0])

    print(tweetclf_compute_metrics("tweetclf", y_pred, y_true))


if __name__ == "__main__":
    main()
