import argparse
import json
import os
from datetime import date

import torch
import wandb

from Benchmarking.Customized_DataLoader import Data_Loader, collate_fn
from Benchmarking.manager import manager_diffusion
from Benchmarking.utils.diffusion_utils import PatchGaussianDiffusion, ConditionalDiffusionPatchModel


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model on custom data')
    parser.add_argument('--split_name', type=str, required=True)
    parser.add_argument('--path_to_folds', type=str, required=True)
    parser.add_argument('--path_to_save', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--feature_size', type=int, default=512)
    parser.add_argument('--target_dim', type=int, default=4096)
    parser.add_argument('--mag', type=str, default='20x')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--loss_type', type=str, default='diff')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.path_to_folds) as f:
        data_dict = json.load(f)

    data_train = Data_Loader(data_dict[args.split_name], phase='train', magnifications=[args.mag], pooling=args.pooling)
    data_val = Data_Loader(data_dict[args.split_name], phase='val', magnifications=[args.mag], pooling=args.pooling)
    data_test = Data_Loader(data_dict[args.split_name], phase='test', magnifications=[args.mag], pooling=args.pooling)

    train_val_loader = {
        'train': torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn),
        'val': torch.utils.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn),
    }
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    teacher = ConditionalDiffusionPatchModel(
        cond_dim=args.feature_size,
        tgt_dim=args.target_dim,
        time_emb_dim=1024,
        hidden_dim=512,
    )
    model = PatchGaussianDiffusion(teacher, num_steps=1000)

    run_name = f"diffusion_{args.split_name}_{args.mag}_{date.today()}"
    run_wb = wandb.init(project=run_name, config=vars(args))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    manager = manager_diffusion(model, device, run_wb, loss_type=args.loss_type)

    best_model, best_optimizer, best_val_loss, best_epoch, train_loss, val_loss = manager.train(train_val_loader, optimizer, epochs=args.epochs)

    output = manager.test(test_loader, best_model)

    os.makedirs(args.path_to_save, exist_ok=True)
    torch.save(output, os.path.join(args.path_to_save, 'output.pt'))
    torch.save({'model': best_model.state_dict()}, os.path.join(args.path_to_save, 'model.pt'))
    run_wb.finish()


if __name__ == '__main__':
    main()
