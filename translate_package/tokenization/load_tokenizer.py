from transformers import BartTokenizerFast
from transformers import T5TokenizerFast
from transformers import AutoTokenizer
import os

def load_tokenizer(tokenizer_name, model, dir_path, file_name, model_name = None):
    
    if model == "nllb":
        
        if not model_name is None:
        
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"The {model}'s tokenizer was successfully loaded")
        
        else:
            
            raise ValueError("For the nllb model you must specify the path to the model!")
    
    if tokenizer_name == "bpe":
        
        tokenizer_path = os.path.join(dir_path, f"{file_name}.json")
        
        if model in ["bart", "lstm"]:
        
            tokenizer = BartTokenizerFast(tokenizer_file=tokenizer_path)
        
            print(f"The Byte Pair Encoding tokenizer was successfully uploaded from {tokenizer_path}")
        
    elif tokenizer_name == "sp":
        
        tokenizer_path = os.path.join(dir_path, f"{file_name}.model")
        
        if model in ['t5', 'mt5']:
            
            tokenizer = T5TokenizerFast(vocab_file=tokenizer_path)
    
            print(f"The Sentence Piece tokenizer was successfully uploaded from {tokenizer_path}")
    
    return tokenizer