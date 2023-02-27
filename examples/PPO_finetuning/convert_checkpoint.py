import torch

from transformers import AutoModel, AutoTokenizer

import sys
import pathlib
path = sys.argv[1]
dir_path = pathlib.Path(path).parent

model = AutoModel.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")
sd = torch.load(path, map_location="cpu") 
sd = {".".join(k.split(".")[2:]):v for k, v in sd.items()}

# allow missing
model.load_state_dict(sd, strict=False)

# save as hf checkpoint 
model.save_pretrained(dir_path)
tokenizer.save_pretrained(dir_path)
