import torch
import numpy as np
from sklearn.metrics import classification_report,f1_score
import torch.nn as nn
import torch.nn.functional as F


def argmax_prediction(pred_y, true_y):
    pred_argmax = torch.argmax(pred_y, dim=1).cpu()
    true_y = true_y.cpu()
    return pred_argmax, true_y


def threshold_prediction(pred_y, true_y):
    pred_y = pred_y > 0.5
    return pred_y, true_y


def metrics_report(pred_y, true_y, label, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = list(label[available_label])
    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)


def metrics_report_for_emo_binary(pred_y, true_y, get_dict=False, multilabel=False):
    if multilabel:
        pred_y, true_y = threshold_prediction(pred_y, true_y)
        available_label = sorted(list(set((pred_y == True).nonzero()[:, -1].tolist() + (true_y == True).nonzero()[:, -1].tolist())))
    else:
        pred_y, true_y = argmax_prediction(pred_y, true_y)
        available_label = sorted(list(set(true_y.tolist() + pred_y.tolist())))

    class_name = ['non-neutral', 'neutral']
    pred_y = [1 if element == 6 else 0 for element in pred_y] # element 6 means neutral.
    true_y = [1 if element == 6 else 0 for element in true_y]

    if get_dict:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4, output_dict=True)
    else:
        return classification_report(true_y, pred_y, target_names=class_name, zero_division=0, digits=4)

def log_metrics(epoch_num,logger, emo_pred_y_list, emo_true_y_list , loss_avg, ece_label_list,ece_prediction,ece_prediction_mask,n_cause,option='train'):
    # 情感
    label_ = np.array(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'])
    logger.info('\n[emotion_reports]\n' +'epoch:'+epoch_num+ metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_))
    report_dict = metrics_report(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), label=label_, get_dict=True)
    acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    # logger.info(f'\nemotion: {option} | loss {loss_avg}\n')
    logger.info(f'\nemotion: accuracy: {acc_emo} | precision: {p_emo} | recall: {r_emo} | f1-score: {f1_emo}\n')


    logger.info('\n' + metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list)))
    report_dict = metrics_report_for_emo_binary(torch.cat(emo_pred_y_list), torch.cat(emo_true_y_list), get_dict=True)
    acc_emo, p_emo, r_emo, f1_emo = report_dict['accuracy'], report_dict['weighted avg']['precision'], report_dict['weighted avg']['recall'], report_dict['weighted avg']['f1-score']
    logger.info(f'\nemotion (binary): {option} | loss {loss_avg}\n')
    
    # 原因
    fscore_ece = f1_score(ece_label_list, ece_prediction,average='macro', \
                          sample_weight=ece_prediction_mask)
    logger.info(f'\ncause (binary): {option} | fscore_ece {fscore_ece}\n')
    cause_reports = classification_report(ece_label_list,
                                    ece_prediction,
                                    target_names=['neg', 'pos'],
                                    sample_weight=ece_prediction_mask,
                                    digits=4)

    # print(cause_reports)
    logger.info('\n[cause_reports]\n'+'epoch:'+epoch_num+cause_reports)



    return fscore_ece,cause_reports,report_dict

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
