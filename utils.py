import torch


def inverse_class_freqs(all_labels: torch.tensor, label_list: list) -> list:

    # NOTE: We assume here that the order of label_list will be used
    # in computing the logits later, and that all_labels contains integers
    # in the range of [0, len(label_list))

    all_labels = list(all_labels)
    counts: torch.FloatTensor = torch.tensor(
        [all_labels.count(i) for i, _ in enumerate(label_list)]
    ).float()

    # We dont know how to do focal loss when there are classes
    # missing in the training data
    assert (counts > 0).all()
    counts = 1 / counts  # Inverse
    counts /= counts.sum()  # Normalize
    ret = counts.tolist()
    return ret


def main():
    all_labels = [0, 0, 1, 1, 1, 2]
    label_list = ["a", "b", "c"]
    print(inverse_class_freqs(all_labels, label_list))


if __name__ == "__main__":
    main()
