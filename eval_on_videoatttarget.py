import torch
from torchvision import transforms
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from model import ModelSpatioTemporal
from dataset import VideoAttTarget_video
from config import *
from utils import imutils, evaluation, misc
from lib.pytorch_convolutional_rnn import convolutional_rnn

import argparse
import os
import numpy as np
from scipy.misc import imresize
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="gpu id")
parser.add_argument("--model_weights", type=str, default='model_videoatttarget.pt', help="model weights")
parser.add_argument("--batch_size", type=int, default=16, help="batch size")
args = parser.parse_args()


def _get_transform():
    transform_list = []
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)


def test():
    transform = _get_transform()

    # Prepare data
    print("Loading Data")
    val_dataset = VideoAttTarget_video(videoattentiontarget_val_data, videoattentiontarget_val_label,
                                        transform=transform, test=True, seq_len_limit=50)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=video_pack_sequences)

    # Define device
    device = torch.device('cuda', args.device)

    # Load model
    num_lstm_layers = 2
    print("Constructing model")
    model = ModelSpatioTemporal(num_lstm_layers = num_lstm_layers)
    model.cuda(device)

    print("Loading weights")
    model_dict = model.state_dict()
    snapshot = torch.load(args.model_weights)
    snapshot = snapshot['model']
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

    print('Evaluation in progress ...')
    model.train(False)
    AUC = []; in_vs_out_groundtruth = []; in_vs_out_pred = []; distance = []
    chunk_size = 3
    with torch.no_grad():
        for batch_val, (img_val, face_val, head_channel_val, gaze_heatmap_val, cont_gaze, inout_label_val, lengths_val) in enumerate(val_loader):
            print('\tprogress = ', batch_val+1, '/', len(val_loader))
            X_pad_data_img, X_pad_sizes = pack_padded_sequence(img_val, lengths_val, batch_first=True)
            X_pad_data_head, _ = pack_padded_sequence(head_channel_val, lengths_val, batch_first=True)
            X_pad_data_face, _ = pack_padded_sequence(face_val, lengths_val, batch_first=True)
            Y_pad_data_cont_gaze, _ = pack_padded_sequence(cont_gaze, lengths_val, batch_first=True)
            Y_pad_data_heatmap, _ = pack_padded_sequence(gaze_heatmap_val, lengths_val, batch_first=True)
            Y_pad_data_inout, _ = pack_padded_sequence(inout_label_val, lengths_val, batch_first=True)

            hx = (torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device),
                  torch.zeros((num_lstm_layers, args.batch_size, 512, 7, 7)).cuda(device)) # (num_layers, batch_size, feature dims)
            last_index = 0
            previous_hx_size = args.batch_size

            for i in range(0, lengths_val[0], chunk_size):
                X_pad_sizes_slice = X_pad_sizes[i:i + chunk_size].cuda(device)
                curr_length = np.sum(X_pad_sizes_slice.cpu().detach().numpy())
                # slice padded data
                X_pad_data_slice_img = X_pad_data_img[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_head = X_pad_data_head[last_index:last_index + curr_length].cuda(device)
                X_pad_data_slice_face = X_pad_data_face[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_cont_gaze = Y_pad_data_cont_gaze[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_heatmap = Y_pad_data_heatmap[last_index:last_index + curr_length].cuda(device)
                Y_pad_data_slice_inout = Y_pad_data_inout[last_index:last_index + curr_length].cuda(device)
                last_index += curr_length

                # detach previous hidden states to stop gradient flow
                prev_hx = (hx[0][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach(),
                           hx[1][:, :min(X_pad_sizes_slice[0], previous_hx_size), :, :, :].detach())

                # forward pass
                deconv, inout_val, hx = model(X_pad_data_slice_img, X_pad_data_slice_head, X_pad_data_slice_face, \
                                                         hidden_scene=prev_hx, batch_sizes=X_pad_sizes_slice)

                for b_i in range(len(Y_pad_data_slice_cont_gaze)):
                    if Y_pad_data_slice_inout[b_i]: # ONLY for 'inside' cases
                        # AUC: area under curve of ROC
                        multi_hot = torch.zeros(output_resolution, output_resolution)  # set the size of the output
                        gaze_x = Y_pad_data_slice_cont_gaze[b_i, 0]
                        gaze_y = Y_pad_data_slice_cont_gaze[b_i, 1]
                        multi_hot = imutils.draw_labelmap(multi_hot, [gaze_x * output_resolution, gaze_y * output_resolution], 3, type='Gaussian')
                        multi_hot = (multi_hot > 0).float() * 1 # make GT heatmap as binary labels
                        multi_hot = misc.to_numpy(multi_hot)

                        scaled_heatmap = imresize(deconv[b_i].squeeze(), (output_resolution, output_resolution), interp = 'bilinear')
                        auc_score = evaluation.auc(scaled_heatmap, multi_hot)
                        AUC.append(auc_score)

                        # distance: L2 distance between ground truth and argmax point
                        pred_x, pred_y = evaluation.argmax_pts(deconv[b_i].squeeze())
                        norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                        dist_score = evaluation.L2_dist(Y_pad_data_slice_cont_gaze[b_i], norm_p).item()
                        distance.append(dist_score)

                # in vs out classification
                in_vs_out_groundtruth.extend(Y_pad_data_slice_inout.cpu().numpy())
                in_vs_out_pred.extend(inout_val.cpu().numpy())

                previous_hx_size = X_pad_sizes_slice[-1]

            try:
                print("\tAUC:{:.4f}"
                      "\tdist:{:.4f}"
                      "\tin vs out AP:{:.4f}".
                      format(torch.mean(torch.tensor(AUC)),
                             torch.mean(torch.tensor(distance)),
                             evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)))
            except:
                pass

    print("Summary ")
    print("\tAUC:{:.4f}"
          "\tdist:{:.4f}"
          "\tin vs out AP:{:.4f}".
          format(torch.mean(torch.tensor(AUC)),
                 torch.mean(torch.tensor(distance)),
                 evaluation.ap(in_vs_out_groundtruth, in_vs_out_pred)))


def video_pack_sequences(in_batch):
    """
    Pad the variable-length input sequences to fixed length
    :param in_batch: the original input batch of sequences generated by pytorch DataLoader
    :return:
        out_batch (list): the padded batch of sequences
    """
    # Get the number of return values from __getitem__ in the Dataset
    num_returns = len(in_batch[0])

    # Sort the batch according to the sequence lengths. This is needed by torch func: pack_padded_sequences
    in_batch.sort(key=lambda x: -x[0].shape[0])
    shapes = [b[0].shape[0] for b in in_batch]

    # Determine the length of the padded inputs
    max_length = shapes[0]

    # Declare the output batch as a list
    out_batch = []
    # For each return value in each sequence, calculate the sequence-wise zero padding
    for r in range(num_returns):
        output_values = []
        lengths = []
        for seq in in_batch:
            values = seq[r]
            seq_size = values.shape[0]
            seq_shape = values.shape[1:]
            lengths.append(seq_size)
            padding = torch.zeros((max_length - seq_size, *seq_shape))
            padded_values = torch.cat((values, padding))
            output_values.append(padded_values)

        out_batch.append(torch.stack(output_values))
    out_batch.append(lengths)

    return out_batch


if __name__ == "__main__":
    test()
