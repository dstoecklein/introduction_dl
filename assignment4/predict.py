import torch
import pandas as pd
from datetime import datetime

def predict_and_save(model, test_loader, path, device):
    results = []

    with torch.no_grad():
        for features, image_id in test_loader:
            features = features.to(device)

            logits = model(features)
            # single class with highest probability. simply retain indices
            _, predictions = torch.max(logits, dim=1)

            # now iterate over each element of the current batch
            for i, features in enumerate(features):
                results.append(
                    [image_id[i].detach().numpy(), predictions[i].cpu().numpy()] # in case the device is set to 'cuda'
                )
    
    df = pd.DataFrame(results, columns =['id', 'classification'])

    filename = 'submission_' + str(type(model).__name__) + datetime.now().strftime('_%Y-%m-%d_%H%M%S.csv')
    df.to_csv(path + filename, index=False, sep=",")
