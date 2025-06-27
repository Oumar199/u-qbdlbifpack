from torch.nn import Module

def get_gradients_mean(model: Module):
    ave_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            ave_grads.append(param.grad.abs().mean().item())
    
    return sum(ave_grads)/len(ave_grads)