import torch
import torch.nn as nn
import torch.nn.functional as F


def _inspect(name, tensor):
    print(name, tensor)


class FocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        alpha: torch.tensor,
        gamma: float,
        reduction: str = "sum",
    ):
        super(FocalLoss, self).__init__()
        assert isinstance(num_classes, int)

        assert alpha.view(-1).shape == (num_classes,)
        assert reduction in ["mean", "none", "sum"]

        self.register_buffer("alpha", alpha.view(1, num_classes))  # (1, C)
        self.num_classes = num_classes
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: torch.tensor, target: torch.tensor):
        """
        input: (N, C)
        target: (N,)
        """
        logSoftmax = nn.LogSoftmax(dim=1)
        softmax = nn.Softmax(dim=1)

        a = self.alpha
        g = self.gamma
        log_y_hat: torch.tensor = logSoftmax(input)
        y_hat: torch.tensor = softmax(input)

        # list(
        #     map(
        #         print,
        #         [
        #             ("y_onehot", y_onehot),
        #             ("a", a),
        #             ("g", g),
        #             ("log_y_hat", log_y_hat),
        #             ("y_hat", y_hat),
        #             ("a", a),
        #             ("res", res),
        #         ],
        #     )
        # )

        per_example_loss = -a * (1 - y_hat) ** g * log_y_hat

        # Select the "right" index for each
        per_example_loss = per_example_loss[range(len(per_example_loss)), target]

        if self.reduction == "none":
            return per_example_loss
        elif self.reduction == "mean":
            norm_factor = self.alpha[0, target].sum()
            return (per_example_loss / norm_factor).sum()
        elif self.reduction == "sum":
            return per_example_loss.sum()


def main():
    input = torch.tensor(
        [
            [100.0, -200.0, -300],
            [100.0, -200.0, -300],
            [-100.0, -200.0, 300],
            [100.0, -200.0, -300],
            [100.0, -200.0, -300],
        ]
    )

    target = torch.tensor([1, 2, 1, 1, 0])
    alpha = torch.tensor([1.0, 2.0, 1])

    alpha /= alpha.sum()
    loss_fct = FocalLoss(num_classes=3, alpha=alpha, gamma=0, reduction="none")
    print("Focal loss, weighted,none", loss_fct(input, target))

    loss_fct = nn.CrossEntropyLoss(weight=alpha, reduction="none")
    print("Cross entropy weighted,none", loss_fct(input, target))

    loss_fct = FocalLoss(num_classes=3, alpha=alpha, gamma=0, reduction="mean")
    print("Focal loss,mean", loss_fct(input, target))

    loss_fct = nn.CrossEntropyLoss(weight=alpha, reduction="mean")
    print("Cross entropy weighted,mean", loss_fct(input, target))

    loss_fct = FocalLoss(num_classes=3, alpha=alpha, gamma=0, reduction="sum")
    print("Focal loss,sum", loss_fct(input, target))

    loss_fct = nn.CrossEntropyLoss(weight=alpha, reduction="sum")
    print("Cross entropy weighted, sum", loss_fct(input, target))


if __name__ == "__main__":
    main()
