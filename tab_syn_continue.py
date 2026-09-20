import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# ===============================
# Tree-based Continual Release
# ===============================
# ===============================
# Table Similarity
# ===============================

def clean_categorical(df, cat_cols):
    df = df.copy()
    for c in cat_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.strip()
            .replace("?", "UNKNOWN")
        )
    return df


def fit_categorical_mapping(df, cat_cols):
    mappings = {}

    for c in cat_cols:
        uniq = df[c].unique().tolist()
        mappings[c] = {
            v: i + 1
            for i, v in enumerate(uniq)
        }

    return mappings


def transform_categorical(df, cat_cols, mappings):

    X_cat = np.zeros(
        (len(df), len(cat_cols)),
        dtype=np.int64)

    for j, c in enumerate(cat_cols):

        mp = mappings[c]

        X_cat[:, j] = df[c].map(
            lambda x: mp.get(x, 0)
        ).to_numpy()

    return X_cat


def normalize_table_torch(X):

    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True) + 1e-6

    return (X - mean) / std


class MixedEncoder(nn.Module):

    def __init__(
            self,
            cat_cardinalities,
            emb_dim=8):

        super().__init__()

        self.embeddings = nn.ModuleList()

        for c in cat_cardinalities:

            d = min(
                emb_dim,
                int(np.ceil(np.sqrt(c + 1)))
            )

            self.embeddings.append(
                nn.Embedding(
                    c + 1,
                    d
                )
            )

    def forward(
            self,
            x_num,
            x_cat):

        embs = []

        for i, emb in enumerate(self.embeddings):
            embs.append(
                emb(x_cat[:, i])
            )

        embs = torch.cat(embs, dim=1)

        return torch.cat(
            [x_num, embs],
            dim=1
        )


@torch.no_grad()
def latent_similarity(
        cand_num,
        priv_num,
        cand_cat,
        priv_cat,
        encoder,
        tau=1.0):

    zc = encoder(
        cand_num,
        cand_cat
    )

    zp = encoder(
        priv_num,
        priv_cat
    )

    diff = (
        zc[:, None, :]
        -
        zp[None, :, :]
    )

    sim = -(
        diff.pow(2).sum(-1)
    ) / (2 * tau ** 2)

    return sim

def latent_similarity_num(
        cand_num,
        priv_num,
        tau=1.0):

    diff = (
        cand_num[:, None, :]
        -
        priv_num[None, :, :]
    )

    sim = -(
        diff.pow(2).sum(dim=2)
    ) / (2 * tau ** 2)

    return sim

class BinaryTreeAggregator:

    def __init__(
            self,
            T,
            dim,
            sigma,
            device):

        self.T = T
        self.dim = dim
        self.sigma = sigma
        self.device = device

        self.tree = {}
        self.time = 0

    def add(self, value):

        self.time += 1

        idx = self.time
        node_sum = value.clone()

        while idx % 2 == 0:

            left = self.tree.pop(idx - 1)
            node_sum += left
            idx //= 2

        noise = (
            torch.randn(
                self.dim,
                device=self.device
            ) * self.sigma
        )

        self.tree[idx] = node_sum + noise

    def prefix_sum(self):

        res = torch.zeros(
            self.dim,
            device=self.device
        )

        idx = self.time

        while idx > 0:

            if idx in self.tree:
                res += self.tree[idx]

            idx -= idx & (-idx)

        return res


def tree_sigma(
        T,
        clip_norm,
        epsilon,
        delta):

    sensitivity = (
            1.0 * clip_norm
            / batch_size
    )

    return (
            sensitivity
            *
            np.sqrt(
                2
                *
                np.log(T)
                *
                np.log(1.25 / delta)
            )
            /
            epsilon
    )


def private_reweighting_streaming(
        candidate_num,
        candidate_cat,
        private_batches_num,
        private_batches_cat,
        encoder,
        epsilon=1.0,
        delta=1e-5,
        lr=1e-3,
        clip_norm=1.0,
        tau=1.0,
        device="cuda"):

    cand_num = torch.tensor(
        candidate_num,
        dtype=torch.float32,
        device=device
    )

    cand_num = normalize_table_torch(
        cand_num
    )

    cand_cat = torch.tensor(
        candidate_cat,
        dtype=torch.long,
        device=device
    )

    N = len(cand_num)

    T = len(private_batches_num)

    sigma = tree_sigma(
        T,
        clip_norm,
        epsilon,
        delta
    )

    tree = BinaryTreeAggregator(
        T,
        N,
        sigma,
        device
    )

    log_w = torch.zeros(
        N,
        device=device
    )

    prev_prefix = torch.zeros(
        N,
        device=device
    )

    for t in range(T):

        priv_num = torch.tensor(
            private_batches_num[t],
            dtype=torch.float32,
            device=device
        )

        priv_num = normalize_table_torch(
            priv_num
        )

        priv_cat = torch.tensor(
            private_batches_cat[t],
            dtype=torch.long,
            device=device
        )

        sim = latent_similarity(
            cand_num,
            priv_num,
            cand_cat,
            priv_cat,
            encoder,
            tau
        )

        # sim = latent_similarity_num(
        #     cand_num,
        #     priv_num,
        #     tau
        # )

        scores = (
            log_w[:, None]
            +
            sim
        )

        probs = torch.softmax(
            scores,
            dim=0
        )

        grad = probs.mean(1)

        norm = torch.norm(grad)

        if norm > clip_norm:
            grad *= (
                clip_norm
                /
                norm
            )

        tree.add(grad)

        prefix = tree.prefix_sum()

        delta_t = (
            prefix
            -
            prev_prefix
        )

        prev_prefix = prefix

        log_w += lr * delta_t

        log_w -= log_w.max()

        if (t + 1) % 10 == 0:
            print(
                f"round {t+1}/{T}"
            )

    w = torch.exp(log_w)
    w /= w.sum()

    return w

# ===============================
# Demo: End-to-End Example
# ===============================
if __name__ == "__main__":
    np.random.seed(42)

    # -------- private table (真实数据) --------
    # 例如：年龄、收入、教育年限
    private_df= pd.read_csv("Real_data/raw_csv/adult.csv")
    candidate_df = pd.read_csv("datasets_llm/adult_100k_good.csv")

    numerical_cols = private_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = private_df.select_dtypes(exclude=np.number).columns.tolist()

    private_df = clean_categorical(private_df, categorical_cols)
    candidate_df = clean_categorical(candidate_df, categorical_cols)

    private_df_sampled = private_df#.sample(n=3000, random_state=42)
    candidate_df = candidate_df#.sample(n=50000, random_state=42)

    batch_size = 100
    epsilon = 0.1

    cat_mappings = fit_categorical_mapping(
        private_df,
        categorical_cols
    )

    private_data_cat = transform_categorical(
        private_df_sampled,
        categorical_cols,
        cat_mappings
    )

    candidate_data_cat = transform_categorical(
        candidate_df,
        categorical_cols,
        cat_mappings
    )

    private_data_num = (
        private_df_sampled[numerical_cols]
        .to_numpy()
        .astype(np.float32)
    )

    candidate_data_num = (
        candidate_df[numerical_cols]
        .to_numpy()
        .astype(np.float32)
    )

    private_batches_num = [
        private_data_num[i:i + batch_size]
        for i in range(
            0,
            len(private_data_num),
            batch_size
        )
    ]

    private_batches_cat = [
        private_data_cat[i:i + batch_size]
        for i in range(
            0,
            len(private_data_cat),
            batch_size
        )
    ]

    cat_cardinalities = [
        len(cat_mappings[c])
        for c in categorical_cols
    ]

    encoder = MixedEncoder(
        cat_cardinalities
    ).cuda()

    weights = private_reweighting_streaming(
        candidate_data_num,
        candidate_data_cat,
        private_batches_num,
        private_batches_cat,
        encoder,
        epsilon=epsilon,
        delta=1e-5,
        lr=0.001,
        clip_norm=1.0,
        tau=1.0,
        device="cuda"
    )

    idx = torch.multinomial(
        weights,
        len(private_df_sampled),
        replacement=True
    ).cpu()

    synthetic_data = candidate_df.iloc[idx]

    # private_df_sampled.to_csv(f'synthetic_data/kdd_private_data_eps{epsilon}.csv')

    synthetic_data.to_csv(f'synthetic_data/adult00_syn_data_eps{epsilon}.csv')


    print("Synthetic data shape:", synthetic_data)


    def to_distribution(arr1, arr2, bins=30):
        # 数值属性：直方图归一化分布
        min_val = min(np.min(arr1), np.min(arr2))
        max_val = max(np.max(arr1), np.max(arr2))
        hist1, _ = np.histogram(arr1, bins=bins, range=(min_val, max_val), density=True)
        hist2, _ = np.histogram(arr2, bins=bins, range=(min_val, max_val), density=True)

        # 避免0导致JS无法计算
        hist1 = hist1 + 1e-9
        hist2 = hist2 + 1e-9
        return hist1 / hist1.sum(), hist2 / hist2.sum()

    for j, col in enumerate(private_df_sampled.columns):
        print(f"{col:15s}")

        # 获取列数据
        private_col = private_df_sampled.iloc[:, j]
        synthetic_col = synthetic_data.iloc[:, j]

        # 判断是否为分类属性
        if private_col.dtype == 'object' or private_col.dtype.name == 'category':
            # 分类属性：条形图 + JS散度
            plt.figure(figsize=(10, 6))

            private_counts = private_col.value_counts(normalize=True)
            synthetic_counts = synthetic_col.value_counts(normalize=True)

            all_categories = private_counts.index.union(synthetic_counts.index)
            private_counts = private_counts.reindex(all_categories, fill_value=1e-9)
            synthetic_counts = synthetic_counts.reindex(all_categories, fill_value=1e-9)

            x = np.arange(len(all_categories))
            width = 0.35

            plt.bar(x - width / 2, private_counts, width, alpha=0.6, label="Private")
            plt.bar(x + width / 2, synthetic_counts, width, alpha=0.6, label="Synthetic")
            plt.xticks(x, all_categories, rotation=45, ha='right')
            plt.xlabel('Categories')
            plt.ylabel('Frequency')
            plt.legend()
            plt.title(f"Feature: {col} (Categorical)")
            plt.tight_layout()
            plt.show()

            # 计算 JS Divergence
            p = private_counts.values
            q = synthetic_counts.values
            jsd = jensenshannon(p, q, base=2)
            print(f"  JS Divergence (JSD): {jsd:.4f}")

        else:
            # 数值属性：直方图 + Wasserstein Distance
            plt.figure(figsize=(10, 6))
            plt.hist(private_col, bins=30, density=True, alpha=0.6, label="Private")
            plt.hist(synthetic_col, bins=30, density=True, alpha=0.6, label="Synthetic")
            plt.legend()
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.title(f"Feature: {col} (Numerical)")
            plt.show()

            # 计算 Wasserstein Distance
            wd = wasserstein_distance(private_col, synthetic_col)
            print(f"  Wasserstein Distance (WD): {wd:.4f}")

            # 数值范围
            try:
                p_min, p_max = private_col.min(), private_col.max()
                s_min, s_max = synthetic_col.min(), synthetic_col.max()
                print(f"  Private range: [{p_min:.4f}, {p_max:.4f}]")
                print(f"  Synthetic range: [{s_min:.4f}, {s_max:.4f}]")
            except Exception as e:
                print(f"  Cannot compute min/max: {e}")

        print("-" * 50)

