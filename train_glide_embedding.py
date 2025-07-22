import argparse
import json
import os
from datetime import date

import torch
import wandb

from Benchmarking.Customized_DataLoader import Glide_DataLoader, glide_collate_fn
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults


def main():
    parser = argparse.ArgumentParser(description="Train GLIDE on custom embeddings")
    parser.add_argument('--split_name', type=str, required=True)
    parser.add_argument('--path_to_folds', type=str, required=True)
    parser.add_argument('--path_to_save', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--text_ctx', type=int, default=128)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.path_to_folds) as f:
        data_dict = json.load(f)

    data_train = Glide_DataLoader(data_dict[args.split_name], phase='train')
    data_val = Glide_DataLoader(data_dict[args.split_name], phase='val')

    train_val_loader = {
        'train': torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, shuffle=True, collate_fn=glide_collate_fn),
        'val': torch.utils.data.DataLoader(data_val, batch_size=args.batch_size, shuffle=False, collate_fn=glide_collate_fn),
    }

    options = model_and_diffusion_defaults()
    options['text_ctx'] = args.text_ctx
    model, diffusion = create_model_and_diffusion(**options)
    model.to(device)

    run = wandb.init(project=f"glide_custom_{date.today()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        for feats, targets, _ in train_val_loader['train']:
            feats = feats.to(device)
            targets = targets.to(device)
            t = torch.randint(0, diffusion.num_timesteps, (feats.size(0),), device=device)
            noise = torch.randn_like(feats)
            x_t = diffusion.q_sample(feats, t, noise)
            optimizer.zero_grad()
            tokens = None
            out = model(x_t, t, embedding=targets)
            loss = torch.nn.functional.mse_loss(out, noise)
            loss.backward()
            optimizer.step()
            run.log({'train_loss': loss.item()})

        model.eval()
        with torch.no_grad():
            for feats, targets, _ in train_val_loader['val']:
                feats = feats.to(device)
                targets = targets.to(device)
                t = torch.randint(0, diffusion.num_timesteps, (feats.size(0),), device=device)
                noise = torch.randn_like(feats)
                x_t = diffusion.q_sample(feats, t, noise)
                out = model(x_t, t, embedding=targets)
                val_loss = torch.nn.functional.mse_loss(out, noise)
                run.log({'val_loss': val_loss.item()})
        torch.save(model.state_dict(), os.path.join(args.path_to_save, f'model_{epoch}.pt'))

    run.finish()


if __name__ == '__main__':
    main()
