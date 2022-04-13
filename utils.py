import torch
import numpy as np


def mse(x, y):
    out = (x - y).pow(2).sum(-1, keepdim=True).mean()
    return out


def gaussian(x, std):
    mean_value = torch.zeros(x.shape)
    std_value = std * torch.ones(x.shape)
    return x + torch.normal(mean_value, std_value).to(x.device)


def rand_ortho(shape, irange):
    A = - irange + 2 * irange * np.random.rand(*shape)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return np.dot(U, np.dot(np.eye(U.shape[1], V.shape[0]), V))


def nll_loss(pred, labels):
    return torch.nn.NLLLoss()(torch.log(pred), labels)


def update_params_rmsprop(updates_all, forward_params, feedback_params, lr_forward, lr_feedback,
                          momentum=0.9, averaging_coeff=0.95, stabilizer=0.001):
    with torch.no_grad():
        for i in range(len(forward_params)):
            param = forward_params[i]
            updates = updates_all[i]
            if len(updates) == 0:
                updates['avg_grad'] = torch.zeros_like(param.data)
                updates['avg_grad_sqr'] = torch.zeros_like(param.data)
                updates['inc'] = torch.zeros_like(param.data)
            avg_grad = updates['avg_grad']
            avg_grad_sqr = updates['avg_grad_sqr']
            inc = updates['inc']
            # 使用这种in-place 函数 updates中的也被更新
            avg_grad = averaging_coeff * avg_grad + (1 - averaging_coeff) * param.grad
            avg_grad_sqr = averaging_coeff * avg_grad_sqr + (1 - averaging_coeff) * param.grad ** 2
            normalized_grad = param.grad / torch.sqrt(avg_grad_sqr - avg_grad ** 2 + stabilizer)
            updated_inc = momentum * inc - lr_forward * normalized_grad
            param[:] = param + updated_inc
            updates['avg_grad'] = avg_grad
            updates['avg_grad_sqr'] = avg_grad_sqr
            updates['inc'] = updated_inc
        for i in range(len(feedback_params)):
            param = feedback_params[i]
            updates = updates_all[i + len(forward_params)]
            if len(updates) == 0:
                updates['avg_grad'] = torch.zeros_like(param.data)
                updates['avg_grad_sqr'] = torch.zeros_like(param.data)
                updates['inc'] = torch.zeros_like(param.data)
            avg_grad = updates['avg_grad']
            avg_grad_sqr = updates['avg_grad_sqr']
            inc = updates['inc']
            # 使用这种in-place 函数 updates中的也被更新
            avg_grad = averaging_coeff * avg_grad + (1 - averaging_coeff) * param.grad
            avg_grad_sqr = averaging_coeff * avg_grad_sqr + (1 - averaging_coeff) * param.grad ** 2
            normalized_grad = param.grad / torch.sqrt(avg_grad_sqr - avg_grad ** 2 + stabilizer)
            updated_inc = momentum * inc - lr_feedback * normalized_grad
            param[:] = param + updated_inc
            updates['avg_grad'] = avg_grad
            updates['avg_grad_sqr'] = avg_grad_sqr
            updates['inc'] = updated_inc
    return updates_all
