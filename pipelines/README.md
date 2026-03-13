# Training Logic
```python
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```