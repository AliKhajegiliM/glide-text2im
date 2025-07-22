import torch
from torch.utils.data import Dataset
import h5py


class Data_Loader(Dataset):
    """Custom dataset for patch and LLM features."""

    def __init__(self, data_dict, phase, transform=None, magnifications=None, super_patch=False, pooling='mean'):
        self.data_dict = data_dict
        self.phase = phase
        self.transform = transform
        self.magnifications = magnifications or ['5x', '10x', '20x']
        self.super_patch = super_patch
        self.pooling = pooling

    def __len__(self):
        return len(self.data_dict['feature_path'][self.phase])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        path_to_dict = self.data_dict['feature_path'][self.phase][idx]
        if path_to_dict.endswith('.h5'):
            f = h5py.File(path_to_dict, 'r')
            feat = torch.tensor(f['features'][self.magnifications[0]]).float()
            hist = torch.tensor(f['histograms'][self.magnifications[0]]).float()
            coord = torch.tensor(f['coords'][self.magnifications[0]]).float()
        elif path_to_dict.endswith('.pt'):
            data = torch.load(path_to_dict)
            feat = data['features'][self.magnifications[0]].float()
            hist = data['histograms'][self.magnifications[0]].float()
            coord = data['coords'][self.magnifications[0]].float()
        else:
            raise NotImplementedError(f"Unsupported file format: {path_to_dict}")

        target_path = self.data_dict['llm_feat'][self.phase][idx]
        target_data = torch.load(target_path)
        hidden = target_data['hidden_states']
        target = hidden.mean(dim=0) if hidden.ndim > 2 else hidden.squeeze(0)

        slide_name = self.data_dict['slide_id'][self.phase][idx]

        if self.pooling == 'None':
            feat = feat.squeeze()
            coord = coord.squeeze()
            hist = hist.squeeze()
            if feat.ndim == 1:
                feat = feat.unsqueeze(0)
                coord = coord.unsqueeze(0)
                hist = hist.unsqueeze(0)
        elif self.pooling == 'mean':
            feat = feat[:, 12, :]
            hist = hist[:, 12, :]
            coord = coord[:, 12, :]
        elif self.pooling == 'all':
            d = feat.shape[2]
            feat = feat.reshape(-1, d)
            hist = hist.reshape(-1, 768)
            coord = coord.reshape(-1, 2)
        else:
            raise NotImplementedError('Invalid pooling type')

        return feat, hist, coord, target, slide_name


def collate_fn(batch):
    feats, hists, coords, targets, slide_names = zip(*batch)

    def pad_tensors(tensor_list):
        max_n = max(tensor.shape[0] for tensor in tensor_list)
        padded = []
        for tensor in tensor_list:
            padding = (0, 0, 0, max_n - tensor.shape[0])
            padded.append(torch.nn.functional.pad(tensor, padding))
        return torch.stack(padded)

    padded_feats = pad_tensors(feats)
    padded_hists = pad_tensors(hists)
    padded_coords = pad_tensors(coords)
    targets = torch.stack(targets)
    slide_names = list(slide_names)
    return padded_feats, padded_hists, padded_coords, targets, slide_names
