# This script adapts some code from
# https://github.com/microsoft/nlp-recipes
import pandas as pd
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from common.article_classification_constants import MAX_SEQ_LEN, SUPPORTED_MODELS


class ArticleClassificationDataProcessor(torch.nn.Module):
    """Class for preprocessing Article classification data.

    Args:
        torch (nn.Module): Inherits from base PyTorch Module
    """

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        to_lower: bool = False,
        batch_size: int = 32,
        num_gpus: int = None,
        cache_dir: str = ".",
    ):
        """ Initialize an ArticleClassificationDataProcessor object

        Args:
            model_name (str, optional): Name of the model.
                Call SequenceClassifier.list_supported_models() to get all supported models.
                Defaults to "bert-base-cased".
            to_lower (bool, optional): Whether to convert all letters to lower case during
                tokenization. This is determined by if a cased model is used.
                Defaults to False, which corresponds to a cased model.
            output_loading_info (bool, optional): Display tokenizer loading info if True.
            batch_size (int, optional): Batch size.
                If more than 1 gpu is used, this would be the batch size per gpu.
                Defaults to 32.
            num_gpus (int, optional): The number of GPUs to be used. Defaults to None.
            cache_dir (str, optional): Directory to cache the tokenizer. Defaults to ".".
        """
        super().__init__()
        self.model_name = model_name
        self.to_lower = to_lower
        self.cache_dir = cache_dir
        self._batch_size = batch_size
        self._num_gpus = num_gpus
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            do_lower_case=self.to_lower,
            cache_dir=self.cache_dir,
            output_loading_info=False,
        )

    @staticmethod
    def text_transform(text: str, tokenizer=None, max_len: int = MAX_SEQ_LEN):
        """
        Text transformation function for sequence classification.
        The function can be passed to a map-style PyTorch DataSet.

        Args:
            text (str): Input text.
            max_len (int, optional): Max sequence length. Defaults to 512.

        Returns:
            tuple: Tuple containing input ids, attention masks, and segment ids.
        """
        if max_len > MAX_SEQ_LEN:
            print("setting max_len to max allowed seq length: {}".format(MAX_SEQ_LEN))
            max_len = MAX_SEQ_LEN
        # truncate and add CLS & SEP markers
        tokens = tokenizer.tokenize(text)[0 : max_len - 2]
        tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

        # get input ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # pad sequence
        input_ids = input_ids + [0] * (max_len - len(input_ids))
        # create input mask
        attention_mask = [min(1, x) for x in input_ids]
        # create segment ids
        token_type_ids = [0] * len(input_ids)

        return input_ids, attention_mask, token_type_ids

    @staticmethod
    def get_inputs(batch, device, model_name, train_mode=True):
        """
        Creates an input dictionary given a model name.

        Args:
            batch (tuple): A tuple containing input ids, attention mask,
                segment ids, and labels tensors.
            device (torch.device): A PyTorch device.
            train_mode (bool, optional): Training mode flag.
                Defaults to True.

        Returns:
            dict: Dictionary containing input ids, segment ids, masks, and labels.
                Labels are only returned when train_mode is True.
        """
        batch = tuple(t.to(device) for t in batch)
        if model_name in SUPPORTED_MODELS:
            if train_mode:
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[3],
                }
            else:
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            # distilbert, bart has no support for segment ids
            if model_name.split("-")[0] not in ["distilbert", "bart"]:
                inputs["token_type_ids"] = batch[2]

            return inputs
        else:
            raise ValueError("Model not supported: {}".format(model_name))

    def create_dataset_from_dataframe(
        self, df: pd.DataFrame, text_col, label_col=None, max_len: int = MAX_SEQ_LEN
    ) -> Dataset:
        """Create a PyTorch Dataset from a pandas Dataframe

        Args:
            df (pd.DataFrame): Dataframe of text to use in sequence classification
            text_col (Union[int, str]): Text column to use in dataframe. Could be an int or str
            label_col (str, optional): Target label to classify. Defaults to None.
            max_len (int, optional): Maximum length to process. Defaults to MAX_SEQ_LEN.

        Returns:
            Dataset: Dataset that will be converted into a Dataloader
        """
        return ArticleClassificationDataSet(
            df,
            text_col,
            label_col,
            transform=ArticleClassificationDataProcessor.text_transform,
            tokenizer=self.tokenizer,
            max_len=max_len,
        )

    def create_dataloader_from_dataset(
        self, dataset: Dataset, shuffle: bool = False, distributed: bool = False,
    ) -> torch.nn.Module:
        """Create a PyTorch DataLoader given a Dataset object.

        Args:
            dataset (torch.utils.data.DataSet): A PyTorch dataset.
            shuffle (bool, optional): If True, a RandomSampler is used. Defaults to False.
            distributed (book, optional): If True, a DistributedSampler is used.
            Defaults to False.

        Returns:
            Module, DataParallel: A PyTorch Module or
                a DataParallel wrapper (when multiple gpus are used).
        """
        if self._num_gpus is None:
            self._num_gpus = torch.cuda.device_count()

        self._batch_size = self._batch_size * max(1, self._num_gpus)

        if distributed:
            sampler = DistributedSampler(dataset)
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        return DataLoader(dataset, sampler=sampler, batch_size=self._batch_size)


class ArticleClassificationDataSet(Dataset):
    """ Create Dataset for single sequence classification tasks """

    def __init__(self, df, text_col, label_col, transform, **transform_args):
        self.df = df
        cols = list(df.columns)
        self.transform = transform
        self.transform_args = transform_args

        if isinstance(text_col, int):
            self.text_col = text_col
        elif isinstance(text_col, str):
            self.text_col = cols.index(text_col)
        else:
            raise TypeError("text_col must be of type int or str")

        if label_col is None:
            self.label_col = None
        elif isinstance(label_col, int):
            self.label_col = label_col
        elif isinstance(label_col, str):
            self.label_col = cols.index(label_col)
        else:
            raise TypeError("label_col must be of type int or str")

    def __getitem__(self, idx):
        input_ids, attention_mask, token_type_ids = self.transform(
            self.df.iloc[idx, self.text_col], **self.transform_args
        )
        if self.label_col is None:
            return tuple(
                [
                    torch.tensor(input_ids, dtype=torch.long),
                    torch.tensor(attention_mask, dtype=torch.long),
                    torch.tensor(token_type_ids, dtype=torch.long),
                ]
            )
        labels = self.df.iloc[idx, self.label_col]
        return tuple(
            [
                torch.tensor(input_ids, dtype=torch.long),  # input_ids
                torch.tensor(attention_mask, dtype=torch.long),  # attention_mask
                torch.tensor(token_type_ids, dtype=torch.long),  # segment ids
                torch.tensor(labels, dtype=torch.long),  # labels
            ]
        )

    def __len__(self):
        return self.df.shape[0]
