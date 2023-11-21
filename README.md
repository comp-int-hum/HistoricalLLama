This project is designed to create a workflow to ocr texts and then pull a Llama LLM to correct them. 


documentation of the llama usage: 

# Example use of Llama-2 on Rockfish (and other environments)

It might be convenient to start `tmux` immediately on logging onto Rockfish, creating a second window (Ctrl-b c), and using the first window for your editor, and the second window for running the model.

To run the model you either need a GPU with sufficient memory (~16GB), or enough RAM (~64GB) to run on the CPU.  On Rockfish (e.g. do this in that second tmux window), you can create a session with enough RAM and then get a terminal on it by running:

```
salloc --mem=64G
interact -t 48:00:00
srun --pty=/bin/bash
```

You'll also need a recent version of Python, and on Rockfish this is set up by running:

```
module load python/3.9.15
```

Now you're ready to set up your virtual environment in a familiar way:

```
git clone https://github.com/comp-int-hum/llama-skeleton.git
cd llama-skeleton
python3 -m venv local
source local/bin/activate
python install -r requirements.txt
```

Finally, you should be able to run the script:

```
python scripts/interactive.py
```

This gives a (very rudimentary) prompt where you can enter some text, and it will write back Llama-2's response, repeating this cycle until you type "exit".  On CPU it will be pretty slow, e.g. it might take 20 seconds to respond.  The code itself is extremely simple though, and should be trivially easy to adapt for arbitrary tasks.
