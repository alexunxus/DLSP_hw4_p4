import numpy as np
from model import resnet20, resnet32, resnet44, resnet56, resnet18, resnet50

from metric import acc

from pipeline import Trainer

from callback import TimeCallback, TimeToAccuracyCallback

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import argparse
import torch.backends.cudnn as cudnn

models = {
    18: resnet18,
    20: resnet20,
    32: resnet32,
    50: resnet50,
    44: resnet44,
    56: resnet56,
}


## accelerate computation
cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

def main(num_layer, GPU_type, toaccuracy=-1):
    
    ## Train with different resnet and GPU type
    epochs = 350
    lr     = 0.01
    moment = 0.9
    batch_size = 128
    nclass     = 10
    checkpoint_path = f"./logs/checkpoint{num_layer}_{GPU_type}.h5"
    log_path        = f"./logs/history{num_layer}_{GPU_type}.json"
    
    # preprocess data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        'valid': transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
    }

    train_dataset = CIFAR10("./cifar10/", train=True  ,download=True, transform=data_transforms['train'])
    valid_dataset = CIFAR10("./cifar10/", train=False ,download=True, transform=data_transforms['valid'])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True,  pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model = models[num_layer]()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=moment)
    scheduler = None
    metrics = [acc]
    criterion = torch.nn.CrossEntropyLoss()

    callbacks = [TimeCallback()]
    if toaccuracy != -1:
        # The training will stop when it reach accuracy 0.92
        callbacks.append(TimeToAccuracyCallback(monitor='val_acc', threshold=toaccuracy))
    
    trainer = Trainer(train_loader, model,criterion= criterion, optim=optimizer, scheduler=scheduler, 
                     nclass = nclass, epochs=epochs, metric_fns=metrics, 
                     val_loader=None if toaccuracy==-1 else valid_loader, 
                     log_path=log_path, callbacks=callbacks)

    trainer.fit()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train CIFAR10 on different GPU.')
    parser.add_argument('--num_layer', type=int,
                        help='an integer in [18|20|32|44|50|56]')
    parser.add_argument('--GPU_type', type=str,
                        help='GPU_type: [K80|P100|A100]')
    parser.add_argument('--toacc', type=float, default=-1,
                        help="train until accuracy(float) reach a specific accuracy")
    args = parser.parse_args()

    if args.num_layer not in [18, 20, 32, 44, 50, 56]:
        raise ValueError(f"{args.num_layer} not in [18|20|32|44|50|56]")
    if args.GPU_type  not in ['K80','P100','A100']:
        raise ValueError(f"{args.GPU_type} not in [K80|P100|A100]")
    if args.toacc < -1 or args.toacc > 1.0:
        raise ValueError(f"Invalid accuracy {args.toacc}")
    main(args.num_layer, args.GPU_type, toaccuracy = args.toacc)
