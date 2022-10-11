class ContrastiveTextTrainer(Trainer):
    def __init__(self, contrastive_loss, head, **kwargs):
        super().__init__(**kwargs)
        
        self.contrastive_loss = contrastive_loss

        self.head = head.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):        
        labels = inputs.get("label")
        negative_labels = inputs.get("label_negative")

        anchor_outputs = model(**inputs['anchor'], output_hidden_states=True)
        positive_outputs = model(**inputs['positive'], output_hidden_states=True)
        negative_outputs = model(**inputs['negative'], output_hidden_states=True)

        embs_layer = -1
        anchor_encodings = self.head(anchor_outputs.hidden_states[embs_layer][:, 0, :])
        positive_encodings = self.head(positive_outputs.hidden_states[embs_layer][:, 0, :])
        negative_encodings = self.head(negative_outputs.hidden_states[embs_layer][:, 0, :])

        loss = self.contrastive_loss(anchor_encodings, positive_encodings, negative_encodings, labels, negative_labels)

        return (loss, positive_outputs) if return_outputs else loss

class ContrastiveImageTrainer(Trainer):
    def __init__(self, contrastive_loss, head, use_negatives=True, **kwargs):
        super().__init__(**kwargs)
        
        self.contrastive_loss = contrastive_loss

        self.head = head.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False):        
        labels = inputs.get("label")
        negative_labels = inputs.get("label_negative")
 
        anchor_outputs = model(**inputs['anchor'], output_hidden_states=True)
        positive_outputs = model(**inputs['positive'], output_hidden_states=True)

        embs_layer = -1
        anchor_encodings = self.head(anchor_outputs.hidden_states[embs_layer][:, 0, :])
        positive_encodings = self.head(positive_outputs.hidden_states[embs_layer][:, 0, :])
        if self.use_negatives:
            negative_outputs = model(**inputs['negative'], output_hidden_states=True)
            negative_encodings = self.head(negative_outputs.hidden_states[embs_layer][:, 0, :])
        else:
            negative_encodings = None
        loss = self.contrastive_loss(anchor_encodings, positive_encodings, negative_encodings, labels, negative_labels)

        return (loss, positive_outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        return super().training_step(model, inputs)
