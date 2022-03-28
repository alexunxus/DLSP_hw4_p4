
from time import time
import torch
import copy
import pandas as pd

def train_model(model, resnet_layers, hardware, dataloaders, 
                dataset_sizes, criterion, optimizer, scheduler, num_epochs=350):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    since = time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    metrics = []
    training_step = 0
    for epoch in range(num_epochs): 
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            epoch_phase_start_time = time()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                step_start_time = time()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()
                        metrics.append({
                            'resnet_layers': resnet_layers,
                            'hardware': hardware,
                            'epoch': epoch,
                            'training_step': training_step,
                            'training_step_loss': loss.item(),
                            'training_step_time': time() - step_start_time
                        })
                        training_step += 1
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_phase_end_time = time()
            print(f'{phase} Loss: {round(epoch_loss, 4)} Acc: {round(epoch_acc.item(), 4)}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time() - since
    print(f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s')
    print(f'Best val Acc: {round(best_acc, 4)}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # set up return structure
    return_df = pd.DataFrame(data=metrics) 
    return model, return_df