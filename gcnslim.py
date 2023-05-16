# -*- coding: utf-8 -*-

r"""
gcnslim
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class GCNSLIM(GeneralRecommender):
    r"""
    GCNSLIM
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GCNSLIM, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(form="coo").astype(np.float32)
        self.X = torch.FloatTensor(self.interaction_matrix.todense()).to(self.device)
        self.LABEL = config["LABEL_FIELD"]

        # load parameters info
        self.latent_dim = config["embedding_size"]  # int type:the embedding size of gcnslim
        self.n_layers = config["n_layers"]  # int type:the layer num of gcnslim
        self.reg_weight = config["reg_weight"]  # float32 type: the weight decay for l2 normalization
        self.alpha = config["alpha"]   # float32 type: the joint optimization weight
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )

        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )

        self.mse_loss = nn.MSELoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_item_similarity = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_item_similarity"]


    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL


    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = []
        #embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            embeddings_list.append(all_embeddings)

        gcnslim_all_embeddings = torch.stack(embeddings_list, dim=1)
        gcnslim_all_embeddings = torch.mean(gcnslim_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            gcnslim_all_embeddings, [self.n_users, self.n_items]
        )

        item_similarity = torch.matmul(item_all_embeddings, item_all_embeddings.transpose(0, 1))

        diag = torch.diag(item_similarity)
        item_similarity_diag = torch.diag_embed(diag)
        item_similarity = item_similarity - item_similarity_diag

        return user_all_embeddings, item_all_embeddings, item_similarity

    def calculate_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_item_similarity is not None:
            self.restore_item_similarity = None

        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label = interaction[self.LABEL]

        user_all_embeddings, item_all_embeddings, item_similarity = self.forward()

        u_history_record = self.X[user]
        i_similarity = item_similarity[item]
        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        item_scores1 = torch.mul(u_history_record, i_similarity).sum(dim=1)
        item_scores2 = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        mse_loss1 = self.mse_loss(item_scores1, label)
        mse_loss2 = self.mse_loss(item_scores2, label)

        # calculate Loss
        u_ego_embeddings = self.user_embedding(user)
        i_ego_embeddings = self.item_embedding(item)

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            i_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mse_loss1 + self.alpha * mse_loss2 + self.reg_weight * reg_loss
        return loss


    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        _, _, item_similarity = self.forward()

        u_history_record = self.X[user]
        i_similarity = item_similarity[item]

        scores = torch.mul(u_history_record, i_similarity).sum(dim=1)
        return scores


    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_item_similarity is None:
            _, _, self.restore_item_similarity = self.forward()

        # get user embedding from storage variable
        u_history_record = self.X[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_history_record, self.restore_item_similarity.transpose(0, 1))
        return scores.view(-1)
