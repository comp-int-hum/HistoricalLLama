import json
from pathlib import Path
import tempfile
import argparse
import os
from typing import List, Literal, Optional, Tuple, TypedDict
import torch
from torch import nn
import llama
from logging import getLogger
from llama import Llama
from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
import torch.distributed


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        dest="checkpoint_path",
        default="/home/sbacker2/corpora/models/llama/llama-2-7b-chat",
        help="Checkpoint path"
    )
    parser.add_argument("--output", dest="output", help="File to save transcript to")
    parser.add_argument(
        "--tokenizer_file",
        dest="tokenizer_file",
        default="/home/sbacker2/corpora/models/llama/tokenizer.model",
        help="Tokenizer model file"
    )
    parser.add_argument("--max_seq_len", dest="max_seq_len", default=1024, type=int)
    parser.add_argument("--max_gen_len", dest="max_gen_len", default=1024, type=int)
    parser.add_argument("--max_batch_size", dest="max_batch_size", default=1, type=int)
    parser.add_argument("--use_gpu", dest="use_gpu", default=False, action="store_true")
    parser.add_argument("--input", dest = input_file)
    args = parser.parse_args()

    if args.use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    
    top_p = 0.9
    temperature = 0.6

    try:
        _, lock = tempfile.mkstemp(prefix="llama")
    
        torch.distributed.init_process_group("gloo", world_size=1, rank=0, init_method="file://{}".format(lock))
        initialize_model_parallel(1)

        checkpoint_file = list(Path(args.checkpoint_path).glob("*.pth"))[0]
        checkpoint = torch.load(checkpoint_file, map_location="cpu")
        with open(Path(args.checkpoint_path) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args = ModelArgs(
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
        )
        tokenizer = Tokenizer(model_path=args.tokenizer_file)

        model_args.vocab_size = tokenizer.n_words
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        llm = Llama(model, tokenizer, device)
        transcript = []
        with open (args.input, "r") as in_file:
            for x in in_file:
                statement = x.read()
                query = None
        
                while True:
                    query = statement.strip()
                    if query == "exit":
                        break
                    transcript.append(query)
                    resp = llm.text_completion([query])[0]
                    print(resp["generation"])
                    transcript.append(resp["generation"])
            if args.output:
                with open(args.output, "wt") as ofd:
                    ofd.write(json.dumps(transcript, indent=2))
    except Exception as e:
        raise e
    finally:
        os.remove(lock)
