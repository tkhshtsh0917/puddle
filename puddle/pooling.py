"""
応用的なPoolingモジュールの実装のコレクションであり、これらを使用することで
従来のPooling手法（MEAN/MAX/CLS）よりも文章の意味をより豊かに捉えられる
可能性があります。

- AttentionPooling: 単一ヘッドのAttentionを用いたPooling
- MultiHeadAttentionPooling: 複数のAttentionヘッドを用いたPooling
- GatedAttentionPooling: Attentionにゲート機構を追加
- SelfAttentionPooling: 学習可能なqueryベクトルを用いた自己注意Pooling
- AdapterAttentionPooling: ボトルネック構造でパラメータを抑えつつ自己注意Pooling
"""

import json
import os

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """
    単一ヘッドのAttention Pooling。
    各トークンの重みを学習可能にし、重要なトークンを強調して文ベクトルを計算します。
    通常のMEAN/CLSよりも文の意味に集中した表現を得やすい。
    """

    def __init__(self, hidden_size, intermediate_size=None, dropout=None):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size

        if dropout is not None and isinstance(dropout, float):
            self.atten = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(intermediate_size, 1),
            )
        else:
            self.atten = nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.Tanh(),
                nn.Linear(intermediate_size, 1),
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        scores = self.atten(token_embeddings).squeeze(-1)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)

        return {"sentence_embedding": pooled}

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        config = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "pytorch_model.bin")))

        return model


class MultiHeadAttentionPooling(nn.Module):
    """
    複数のAttentionヘッドを用いたPooling。
    文の異なる側面を捉えることができ、リッチな表現が得られます。
    headsの出力を結合し、線形変換して文ベクトルに変換します。
    """

    def __init__(self, hidden_size, intermediate_size=None, dropout=None, num_heads=4):
        super().__init__()

        if intermediate_size is None:
            intermediate_size = hidden_size

        self.heads = nn.ModuleList(
            [
                AttentionPooling(hidden_size, intermediate_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(num_heads * intermediate_size, hidden_size)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_heads = num_heads

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        head_outputs = [
            head(token_embeddings, attention_mask)["sentence_embedding"]
            for head in self.heads
        ]
        concat = torch.cat(head_outputs, dim=-1)
        pooled = self.proj(concat)

        return {"sentence_embedding": pooled}

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        config = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "dropout": self.dropout,
            "num_heads": self.num_heads,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "pytorch_model.bin")))

        return model


class GatedAttentionPooling(nn.Module):
    """
    Attentionに加え、トークンごとのゲート機構（Sigmoid）を導入。
    ノイズとなるトークンを抑制し、文の意味に寄与する部分だけを強調できます。
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.atten = nn.Linear(hidden_size, 1)
        self.gate = nn.Linear(hidden_size, 1)

        self.hidden_size = hidden_size

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        score = torch.tanh(self.atten(token_embeddings)).squeeze(-1)
        gate = torch.sigmoid(self.gate(token_embeddings)).squeeze(-1)
        score = score * gate  # combine gating with relevance

        if attention_mask is not None:
            score = score.masked_fill(attention_mask == 0, -1e9)

        weights = torch.softmax(score, dim=-1).unsqueeze(-1)
        pooled = torch.sum(weights * token_embeddings, dim=1)

        return {"sentence_embedding": pooled}

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        config = {"hidden_size": self.hidden_size}
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "pytorch_model.bin")))

        return model


class SelfAttentionPooling(nn.Module):
    """
    学習可能なqueryベクトルを用いた自己注意Pooling。
    特別なトークン（[CLS]）を使わず、全体を代表する重み付けベクトルを学習できます。
    """

    def __init__(self, hidden_size):
        super().__init__()

        self.query = nn.Parameter(torch.randn(hidden_size))

        self.hidden_size = hidden_size

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        scores = torch.matmul(token_embeddings, self.query)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(token_embeddings * weights.unsqueeze(-1), dim=1)

        return {"sentence_embedding": pooled}

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        config = {"hidden_size": self.hidden_size}
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "pytorch_model.bin")))

        return model


class AdapterAttentionPooling(nn.Module):
    """
    ボトルネック構造でパラメータを抑えつつ、自己注意に基づくPoolingを行う。
    Adapter構造に似た設計で、追加学習がしやすく文表現の強化にも有効。
    """

    def __init__(self, hidden_size, bottleneck=64, dropout=None):
        super().__init__()

        self.down_proj = nn.Linear(hidden_size, bottleneck)
        if dropout is not None and isinstance(dropout, float):
            self.atten = nn.Sequential(
                nn.Linear(bottleneck, bottleneck),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck, 1),
            )
        else:
            self.atten = nn.Sequential(
                nn.Linear(bottleneck, bottleneck),
                nn.Tanh(),
                nn.Linear(bottleneck, 1),
            )
        self.up_proj = nn.Linear(bottleneck, hidden_size)

        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.dropout = dropout

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        reduced = self.down_proj(token_embeddings)
        scores = self.atten(reduced).squeeze(-1)

        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)

        weights = torch.softmax(scores, dim=-1)
        pooled = self.up_proj(torch.sum(reduced * weights.unsqueeze(-1), dim=1))

        return {"sentence_embedding": pooled}

    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        config = {
            "hidden_size": self.hidden_size,
            "bottleneck": self.bottleneck,
            "dropout": self.dropout,
        }
        with open(os.path.join(output_path, "config.json"), "w") as f:
            json.dump(config, f)

        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @classmethod
    def load(cls, input_path):
        with open(os.path.join(input_path, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model.load_state_dict(torch.load(os.path.join(input_path, "pytorch_model.bin")))

        return model
