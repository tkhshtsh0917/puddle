# puddle

This library, named *puddle*, is a collection of more applied learnable pooling layer implementations other than the widely used CLS/MEAN/MAX pooling.

When used in conjunction with any pre-trained model, they have the potential to generate embedded vectors that capture the meaning of sentences more highly than the widely used CLS/MEAN/MAX pooling.

## Usage

```python
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses, models
from datasets import Dataset

from puddle import AttentionPooling

data = {
    "anchor": ["anchor_text", ...],
    "positive": ["positive_text", ...],
    "negative": ["negative_text", ...]
}
train_dataset = Dataset.from_dict(data)

transformer_layer = models.Transformer("your-awesome-model")
hidden_size: int = transformer_layer.get_word_embedding_dimension()
pooling_layer = AttentionPooling(
    hidden_size=hidden_size,
    intermediate_size=hidden_size,
)

model = SentenceTransformer(modules=[transformer_layer, pooling_layer])

triplet_loss = losses.TripletLoss(model=model)
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=triplet_loss,
)

trainer.train()
model.save_pretrained("your-awesome-model-with-attention-pooling")
```
