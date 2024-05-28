import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import yaml
import time
import os
import matplotlib.pyplot as plt
from dataloader import generate_dataloader
from torchcontrib.optim import SWA
from torch import optim
from dependency import *
from model import 7PGCN
from tensorboardX import SummaryWriter
from utils import Logger, adjust_learning_rate, CraateLogger, create_cosine_learing_schdule, encode_test_label, set_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, confusion_matrix

import logging
logging.basicConfig(filename='./logfile_atypical.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

writer = SummaryWriter("./tensorboard/")

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def calculate_roc_and_metrics_diag(logits, labels, threshold=0.6):
    probabilities = sigmoid(logits)
    predictions = (probabilities.detach().cpu().numpy() > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels.detach().cpu().numpy(), predictions).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0

    fpr, tpr, _ = roc_curve(labels.detach().cpu().numpy(), probabilities)
    auc_score = roc_auc_score(labels.detach().cpu().numpy(), probabilities)

    metrics = {
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc_score,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision
    }

    return metrics

def calculate_roc_and_metrics(logits, labels, threshold=0.5):
    probabilities = torch.sigmoid(logits)
    num_labels = labels.shape[1]-1

    all_predictions = np.zeros_like(labels.detach().cpu().numpy())
    all_predictions[:, -1] = labels[:, 1].cpu().numpy()

    metrics_per_label = {}
    total_auc = 0
    total_sensitivity = 0
    total_specificity = 0
    total_precision = 0

    # temp_data = probabilities.detach().cpu().numpy()
    # np.savetxt("output.csv", temp_data, delimiter=",", fmt='%s')
    
    for label_idx in range(num_labels):
        probabilities_label = probabilities[:, label_idx].detach().cpu().numpy()
        labels_label = labels[:, label_idx].detach().cpu().numpy()

        # Apply the threshold to get binary predictions
        predictions = (probabilities_label > threshold).astype(int)
        all_predictions[:, label_idx] = predictions
        
        # Calculate confusion matrix and extract TP, FP, FN, TN
        tn, fp, fn, tp = confusion_matrix(labels_label, predictions).ravel()

        # Compute sensitivity, specificity, and precision
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0

        fpr, tpr, _ = roc_curve(labels_label, probabilities_label)
        auc = roc_auc_score(labels_label, probabilities_label)

        total_auc += auc
        total_sensitivity += sensitivity
        total_specificity += specificity
        total_precision += precision

        metrics_per_label[label_idx] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'auc': auc
        }
    np.savetxt("predictions.csv", all_predictions, delimiter=",", fmt='%d')

    avg_auc = total_auc / num_labels
    avg_sensitivity = total_sensitivity / num_labels
    avg_specificity = total_specificity / num_labels
    avg_precision = total_precision / num_labels

    return metrics_per_label, avg_auc, avg_sensitivity, avg_specificity, avg_precision

def validation(net, diag_weights=None, val_dataloader=None):
    net.eval()
    val_loss = 0
    all_logits_derm = []
    all_logits_clic = []
    all_logits_fusion = []
    all_labels = []
    
    all_logits_weighted_derm = []
    all_logits_weighted_clic = []
    all_logits_weighted_fusion = []
    
    for index, (clinic_image, derm_image, label) in enumerate(val_dataloader):
        clinic_image = clinic_image.cuda()
        derm_image = derm_image.cuda()
        # meta_data = meta_data.cuda()
        diag_scale = 2
        diagnosis_label = label[0].long().cuda()
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()
        labels = [diagnosis_label, pn_label, str_label, pig_label,
                  rs_label, dag_label, bwv_label, vs_label]

        with torch.no_grad():
            [(logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, logit_vs_derm, logits_combined_derm),
             (logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic,
              logit_bwv_clic, logit_vs_clic, logits_combined_clic),
             (logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, logit_vs_fusion, logits_combined_fusion)] = net((clinic_image, derm_image))
            
            if diag_weights is not None:
                weighted_derm_logits = torch.sigmoid(logits_combined_derm * diag_weights)
                weighted_clic_logits = torch.sigmoid(logits_combined_clic * diag_weights)
                weighted_fusion_logits = torch.sigmoid(logits_combined_fusion * diag_weights)

            loss_fusion = torch.true_divide(
                diag_scale*net.criterionBCE(weighted_fusion_logits, diagnosis_label)
                + net.criterionBCE(logit_pn_fusion, pn_label)
                + net.criterionBCE(logit_str_fusion, str_label)
                + net.criterionBCE(logit_pig_fusion, pig_label)
                + net.criterionBCE(logit_rs_fusion, rs_label)
                + net.criterionBCE(logit_dag_fusion, dag_label)
                + net.criterionBCE(logit_bwv_fusion, bwv_label)
                + net.criterionBCE(logit_vs_fusion, vs_label), 7)

            loss_clic = torch.true_divide(
                diag_scale*net.criterionBCE(weighted_clic_logits, diagnosis_label)
                + net.criterionBCE(logit_pn_clic, pn_label)
                + net.criterionBCE(logit_str_clic, str_label)
                + net.criterionBCE(logit_pig_clic, pig_label)
                + net.criterionBCE(logit_rs_clic, rs_label)
                + net.criterionBCE(logit_dag_clic, dag_label)
                + net.criterionBCE(logit_bwv_clic, bwv_label)
                + net.criterionBCE(logit_vs_clic, vs_label), 7)

            loss_derm = torch.true_divide(
                diag_scale*net.criterionBCE(weighted_derm_logits, diagnosis_label)
                + net.criterionBCE(logit_pn_derm, pn_label)
                + net.criterionBCE(logit_str_derm, str_label)
                + net.criterionBCE(logit_pig_derm, pig_label)
                + net.criterionBCE(logit_rs_derm, rs_label)
                + net.criterionBCE(logit_dag_derm, dag_label)
                + net.criterionBCE(logit_bwv_derm, bwv_label)
                + net.criterionBCE(logit_vs_derm, vs_label), 7)

            loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33

        val_loss += loss.item()
        logits_derm = torch.cat([logit_pn_derm, logit_str_derm, logit_pig_derm,
                                logit_rs_derm, logit_dag_derm, logit_bwv_derm, logit_vs_derm], dim=1)
        logits_clic = torch.cat([logit_pn_clic, logit_str_clic, logit_pig_clic,
                                logit_rs_clic, logit_dag_clic, logit_bwv_clic, logit_vs_clic], dim=1)
        logits_fusion = torch.cat([logit_pn_fusion, logit_str_fusion, logit_pig_fusion,
                                  logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, logit_vs_fusion], dim=1)

        all_logits_derm.append(logits_derm)
        all_logits_clic.append(logits_clic)
        all_logits_fusion.append(logits_fusion)

        # Concatenate labels for different tasks and add to list
        concatenated_labels = torch.stack(labels, dim=1)
        all_labels.append(concatenated_labels)

        all_logits_weighted_derm.append(weighted_derm_logits.detach().cpu())
        all_logits_weighted_clic.append(weighted_clic_logits.detach().cpu())
        all_logits_weighted_fusion.append(weighted_fusion_logits.detach().cpu())
        # ... [rest of the loop, including loss calculation and optimization]

    # Concatenate all logits and labels after the loop
    all_logits_derm = torch.cat(all_logits_derm, dim=0)
    all_logits_clic = torch.cat(all_logits_clic, dim=0)
    all_logits_fusion = torch.cat(all_logits_fusion, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    all_logits_weighted_derm = torch.cat(all_logits_weighted_derm, dim=0)
    all_logits_weighted_clic = torch.cat(all_logits_weighted_clic, dim=0)
    all_logits_weighted_fusion = torch.cat(all_logits_weighted_fusion, dim=0)

    val_loss = val_loss / (index + 1)

    metrics_per_label_derm, avg_auc_derm, avg_sensitivities_derm, avg_specificities_derm, avg_precision_derm = calculate_roc_and_metrics(
        all_logits_derm, all_labels[:, 1:8])
    metrics_per_label_clic, avg_auc_clic, avg_sensitivities_clic, avg_specificities_clic,avg_precision_clic = calculate_roc_and_metrics(
        all_logits_clic, all_labels[:, 1:8])
    metrics_per_label_fusion, avg_auc_fusion, avg_sensitivities_fusion, avg_specificities_fusion, avg_precision_fusion = calculate_roc_and_metrics(
        all_logits_fusion, all_labels[:, 1:8])
    metrics_per_label_fusion, avg_auc_fusion, avg_sensitivities_fusion, avg_specificities_fusion, avg_precision_fusion = calculate_roc_and_metrics(
        all_logits_fusion, all_labels[:, 0:8])
    metrics_derm = calculate_roc_and_metrics_diag(all_logits_weighted_derm, all_labels[:, 0])
    metrics_clic = calculate_roc_and_metrics_diag(all_logits_weighted_clic, all_labels[:, 0])
    metrics_fusion = calculate_roc_and_metrics_diag(all_logits_weighted_fusion, all_labels[:, 0])
    
    # print("fpr:", metrics_fusion["fpr"])
    # print("tpr:", metrics_fusion["tpr"])
    
    # sensitivity = avg_sensitivities_fusion
    # specificity = avg_specificities_fusion
    
    # # Typically, the x-axis for such a curve would be 1-specificity
    # one_minus_specificity = 1 - specificity

    # # Creating the plot
    # plt.figure(figsize=(8, 6))
    # plt.plot(one_minus_specificity, sensitivity, marker='o')

    # # Adding title and labels
    # plt.title('Sensitivity and Specificity Curve')
    # plt.xlabel('1-Specificity (False Positive Rate)')
    # plt.ylabel('Sensitivity (True Positive Rate)')

    # # Show grid
    # plt.grid(True)

    # image_path = './sensitivity_specificity_curve.png'
    # plt.savefig(image_path)
    
    for idx in range(0, 7):
        logging.info("{}:{}".format(idx, metrics_per_label_fusion[idx]["auc"]))
        # print("sensitivity: ", metrics_per_label_fusion[idx]["sensitivity"])
        # print("specificity:", metrics_per_label_fusion[idx]["specificity"])
        # print("precision:", metrics_per_label_fusion[idx]["precision"])
    
    logging.info("-------------------Validation: ROC AUC Scores:")
    logging.info(f"-------------------AUC Derm: {avg_auc_derm:.4f}--clic:{avg_auc_clic:.4f}-- Fusion:{avg_auc_fusion:.4f}")
    logging.info(f"-------------------Diagnogis AUC Derm: {metrics_derm['auc']:.4f}--clic:{metrics_clic['auc']:.4f}-- Fusion:{metrics_fusion['auc']:.4f}")
    

    final_sensitivity = (avg_sensitivities_fusion*7 + metrics_fusion['sensitivity'])/8
    final_specificity = (avg_specificities_fusion*7 + metrics_fusion['specificity'])/8
    final_precision = (avg_precision_fusion*7 + metrics_fusion['precision'])/8
    final_auc = (avg_auc_fusion*7 + metrics_fusion['auc'])/8
    logging.info(metrics_fusion)
    logging.info(f"------------------------------------------------final average AUC: {final_auc}")
    logging.info(f"------------------------------------------------final average precision: {final_precision}")
    logging.info(f"------------------------------------------------final average sensitivity: {final_sensitivity}")
    logging.info(f"------------------------------------------------final average specificity: {final_specificity}")
    logging.info(f"{metrics_fusion['sensitivity']}, {metrics_fusion['specificity']}, {metrics_fusion['precision']}")
    logging.info('weights:', diag_weights)

    return avg_auc_fusion
    # return metrics_fusion['auc']


def run_check(net, optimizer):
    # Load Checkpoint

    checkpoint_path = "./auc84_diag88/checkpoint/"
    net.load_state_dict(torch.load(os.path.join(checkpoint_path, "best_AUC_model.pth")))
    diag_params = torch.load(os.path.join(checkpoint_path, "diag_weights.pth"))
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    # validation mode
    val_AUC = validation(
        net, diag_params, val_dataloader)

    print("average AUC: {}".format(val_AUC))


def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = load_config('config.yaml')
    mode = config['mode']
    model_name = config['model_name']
    shape = tuple(config['shape'])
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    data_mode = config['data_mode']
    deterministic = config['deterministic']
    if deterministic:
        random_seeds = config['random_seeds'][data_mode]
    rounds = config['rounds']
    lr = config['lr']
    epochs = config['epochs']
    swa_epoch = config['swa_epoch']

    train_dataloader, val_dataloader = generate_dataloader(
        shape, batch_size, num_workers, data_mode)

    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)

        log, out_dir = CreateLogger(mode, model_name, i, data_mode)
        net = 7PGCN().cuda()  # Ensure 7PGCN is defined elsewhere
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        opt = SWA(optimizer)
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        # start = time.time()
        run_check(net, optimizer)
        # end = time.time()
        # processing_time_minutes = (end - start) / 60
        # print("Inference time:", processing_time_minutes, "minutes")

