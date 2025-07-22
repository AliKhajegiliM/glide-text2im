import copy
import time
import torch
import torch.nn as nn


class manager_diffusion:
    def __init__(self, model, device, run_wb, loss_type='diff'):
        self.model = model
        self.device = device
        self.run_wb = run_wb
        if device == 'cuda':
            self.model.cuda()

        if loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_type == 'MAE':
            self.criterion = nn.L1Loss()
        elif loss_type == 'Huber':
            self.criterion = nn.SmoothL1Loss(beta=1.0)
        elif loss_type == 'KL':
            self.criterion = nn.KLDivLoss()
        elif loss_type == 'CE':
            self.criterion = nn.CrossEntropyLoss()
        else:  # default for diffusion
            self.criterion = nn.MSELoss()

    def train(self, train_val_loader, optimizer, epochs, scheduler_flag=False):
        self.model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.train_loss = {}
        self.val_loss = {}
        self.best_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(epochs):
            self.train_loss[f'epoch-{epoch}'] = []
            self.val_loss[f'epoch-{epoch}'] = []
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                for step, (feat, hist, coord, target, slide) in enumerate(train_val_loader[phase]):
                    feat = feat.to(self.device)
                    target = target.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        loss = self.model(target, feat)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * feat.size(0)
                    if phase == 'train':
                        self.train_loss[f'epoch-{epoch}'].append(loss.item())
                    else:
                        self.val_loss[f'epoch-{epoch}'].append(loss.item())
                    self.run_wb.log({'step': step, f'{phase}_loss': loss.item()})

                epoch_loss = running_loss / len(train_val_loader[phase].dataset)
                self.run_wb.log({'epoch': epoch, f'{phase}_loss': epoch_loss})
                if phase == 'val' and running_loss < self.best_val_loss:
                    self.best_val_loss = running_loss
                    self.best_epoch = epoch
                    self.best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.best_optimizer = copy.deepcopy(self.optimizer)
            if scheduler_flag:
                self.scheduler.step()
        self.model.load_state_dict(self.best_model_wts)
        return self.model, self.best_optimizer, self.best_val_loss, self.best_epoch, self.train_loss, self.val_loss

    def test(self, test_loader, best_model):
        best_model.to(self.device)
        best_model.eval()
        running_loss = 0.0
        start = time.time()
        for feat, hist, coord, targets, slide in test_loader:
            feat = feat.to(self.device)
            targets = targets.to(self.device)
            outputs = best_model.sample(feat)
            loss = self.criterion(outputs, targets)
            running_loss += loss.item() * feat.size(0)
        duration = time.time() - start
        avg_time = float(duration / len(test_loader))
        test_loss = running_loss / len(test_loader.dataset)
        self.run_wb.log({'test_loss': test_loss})
        return {'loss': test_loss, 'average_time': avg_time}
