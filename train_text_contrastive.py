import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from mizan_embedder.model import MizanEmbeddingModel
from mizan_embedder.data import TextPairDataset, make_collate_fn
from mizan_vector.losses import MizanContrastiveLoss

def main():
    model_name="distilbert-base-uncased"
    device="cuda" if torch.cuda.is_available() else "cpu"

    pairs=[
        ("what is mizan?","mizan is a scale-aware similarity function",1),
        ("cosine similarity","apples are fruit",0),
        ("who made mizan?","Ahsan Shaokat invented the Mizan function",1),
    ]

    tok=AutoTokenizer.from_pretrained(model_name)
    ds=TextPairDataset(pairs)
    loader=DataLoader(ds,batch_size=2,shuffle=True,
                      collate_fn=make_collate_fn(tok))

    model=MizanEmbeddingModel(model_name,384,"mean").to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5)
    loss_fn=MizanContrastiveLoss()

    for epoch in range(3):
        total=0
        for e1,e2,lbl in loader:
            e1={k:v.to(device) for k,v in e1.items()}
            e2={k:v.to(device) for k,v in e2.items()}
            lbl=lbl.to(device)
            v1=model(**e1); v2=model(**e2)
            loss=loss_fn(v1,v2,lbl)
            opt.zero_grad(); loss.backward(); opt.step()
            total+=loss.item()
        print(f"epoch {epoch+1} loss={total/len(loader):.4f}")

if __name__=="__main__":
    main()
