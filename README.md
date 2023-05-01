# Contrastive Transformers

A huggingface trainer that implements the contrastive learning process. The ContrastiveTrainer class allows the usage of a custom loss and a customized projection head architecture. The anchor, positive, and negative pairs can be passed with the training dataset or they can be extracted from the labels in each training batch.

```python
contrastive_head = nn.Sequential(
    nn.Linear(768, 768 // 2),
    nn.ReLU(),
    nn.Linear(768 // 2, 768 // 4),
    nn.ReLU(),
    nn.Linear(768 // 4, 768 // 8),
)

ct_loss = SupConLoss(0.2)

def loss_adapter(anchor_encodings, 
                 positive_encodings, 
                 negative_encodings, 
                 labels, 
                 negative_labels, 
                 anchor_outputs, positive_outputs, negative_outputs):
    contrastive_loss = (
        ct_loss(anchor_encodings, positive_encodings, labels) + 
        ct_loss(positive_encodings, anchor_encodings, labels)
    )

    return contrastive_loss

trainer = ContrastiveTrainer(
    loss=loss_adapter,
    head=contrastive_head,
    use_negatives=False,
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train,
    tokenizer=feature_extractor
)

trainer.train()
```
