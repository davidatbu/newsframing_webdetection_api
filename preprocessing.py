"""
Note on something not intuitive, but necessary below.

The ids in the tsvs in the folds/ directory correspond to the "ImageID" column in annotated.tsv.
The ids in web_detect.csv correspond to the "ID" column in annotated.tsv. 
This is the exact reverse of what one would expect.
"""

import logging
import typing as T
import csv
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

data_d: Path = Path(
    "/projectnb/llamagrp/davidat/projects/newsframing/images/webdetectionclf/data"
)
ready_d = data_d / "ready"
raw_d: Path = data_d / "raw"
annotated_tsv = raw_d / "annotated.tsv"
web_detect_csv = raw_d / "web_detect.csv"
folds_d = raw_d / "folds"

relevant_subset_d = ready_d / "relevant"
all_data_d = ready_d / "all_data"
all_data_with_webdetect_for_relevant_d = (
    ready_d / "all_data_with_webdetect_for_relevant_d"
)

logger = logging.getLogger()


def read_fold_dfs() -> T.Dict[str, T.Tuple[pd.DataFrame, pd.DataFrame]]:
    result: T.Dict[str, T.Tuple[pd.DataFrame, pd.DataFrame]] = {}
    for fold_d in folds_d.iterdir():
        with open(fold_d / "train.tsv") as f:
            train_df: pd.DataFrame = pd.read_csv(
                f, delimiter="\t", header=None,
            )
        with open(fold_d / "test.tsv") as f:
            dev_df: pd.DataFrame = pd.read_csv(
                f, delimiter="\t", header=0,
            )
        train_df = train_df.rename(columns={0: "ImageID", 1: "news_title"})[
            ["ImageID", "news_title"]
        ].set_index(["ImageID", "news_title"])
        dev_df = dev_df.rename(
            columns={"guid": "ImageID", "text_a": "news_title"}
        ).set_index(["ImageID", "news_title"])
        logger.info(
            "Fold {}: read {} train  and {} test .".format(
                fold_d.name, train_df.shape, dev_df.shape
            )
        )
        result[fold_d.name] = (train_df, dev_df)
    return result


def join_webdetect_and_anno(
    relevant_only: bool, add_webdetect_for_relevant_only: bool
) -> pd.DataFrame:
    with open(annotated_tsv) as f:
        annotated_df: pd.DataFrame = pd.read_csv(
            f, delimiter="\t", quoting=csv.QUOTE_NONE
        )
        if relevant_only:
            annotated_df = annotated_df[annotated_df["V3relevance"] == 1]
    with open(web_detect_csv) as f:
        web_detect_df: pd.DataFrame = pd.read_csv(f, header=None)

    # Select the first, and then the even(or zero-indexed odd) rows from web_detect_df
    # strip the .jpg from the first column
    # Join the even rows with a comma
    entities_df: pd.DataFrame = web_detect_df.iloc[
        :, [0] + [i for i in range(1, len(web_detect_df.columns), 2)]
    ].apply(
        lambda row: (  # tuple
            int(row[0].strip(".jpg")),  # first element is ID
            ", ".join(
                [i.strip() for i in row[1:].tolist() if i == i and i.strip()]
            ),  # Second element is ", " joined list of entities
        ),
        axis=1,
        result_type="expand",
    )

    # Prepare for oing
    entities_df.rename(columns={0: "ID", 1: "WebDetectEntities"}, inplace=True)
    entities_df.set_index("ID", inplace=True)

    joined_df = annotated_df.join(entities_df, on="ID", how="inner")

    if add_webdetect_for_relevant_only:
        joined_df.loc[(joined_df["V3relevance"] != 1), "WebDetectEntities"] = ""
    return joined_df


def df_diff(df1, df2, which=None):
    """Find rows which are different between two DataFrames."""
    comparison_df = df1.merge(df2, indicator=True, how="outer")
    diff_df = comparison_df[comparison_df["_merge"] == which]
    diff_df = diff_df.drop("_merge", axis="columns")
    return diff_df


def main():
    parser = ArgumentParser()
    parser.add_argument("-r", "--relevant_only", action="store_true")
    parser.add_argument("--add_webdetect_for_relevant_only", action="store_true")
    args = parser.parse_args()

    assert not (args.add_webdetect_for_relevant_only and args.relevant_only)

    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    joined_df = join_webdetect_and_anno(
        args.relevant_only, args.add_webdetect_for_relevant_only
    )

    # Make sure we have no duplicates
    id_title_counts: pd.Series = joined_df[["ImageID", "news_title"]].apply(
        lambda row: (row[0], row[1]), axis=1
    ).value_counts()
    if len(set(id_title_counts.tolist())) != 1:
        logging.error(
            "There are duplicates in the 'primary key': {}".format(id_title_counts)
        )

    dfs_per_fold = read_fold_dfs()
    ready_d.mkdir(exist_ok=True)

    if args.relevant_only:
        out_d = relevant_subset_d
    elif args.add_webdetect_for_relevant_only:
        out_d = all_data_with_webdetect_for_relevant_d
    else:
        out_d = all_data_d
    out_d.mkdir(exist_ok=True)

    for (fold_name, (train_fold_df, dev_fold_df),) in dfs_per_fold.items():
        fold_d = out_d / fold_name
        fold_d.mkdir(exist_ok=True)
        train_df: pd.DataFrame = joined_df.join(
            train_fold_df, on=("ImageID", "news_title"), how="inner"
        )
        dev_df: pd.DataFrame = joined_df.join(
            dev_fold_df, on=("ImageID", "news_title"), how="inner"
        )
        with open(fold_d / "train.tsv", "w") as f:
            train_df.to_csv(f, sep="\t", index=False)
        with open(fold_d / "dev.tsv", "w") as f:
            dev_df.to_csv(f, sep="\t", index=False)
        logger.info(
            "Fold {}: Wrote train df of shape {} and dev df of shape {}.".format(
                fold_d.name, train_df.shape, dev_df.shape
            )
        )

        remaining_rows: pd.DataFrame = df_diff(
            df_diff(joined_df, train_df, which="left_only"), dev_df, which="left_only"
        )[["ImageID", "news_title"]]
        if len(remaining_rows) > 0:
            logging.error(
                "The following rows were not written to train OR dev:\n{}".format(
                    remaining_rows
                )
            )


if __name__ == "__main__":
    main()
