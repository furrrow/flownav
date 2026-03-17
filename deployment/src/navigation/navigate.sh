#!/bin/bash

# Create a new tmux session
session_name="flownav_$(date +%s)"
tmux new-session -d -s $session_name

tmux set-option -g mouse on


# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves


# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 0
tmux send-keys "source /workspace/prune/setup.bash" Enter
#tmux send-keys "source .venv/bin/activate" Enter
#tmux send-keys "python navigate.py $@"
tmux send-keys "uv run deployment/src/navigation/navigate.py --dir antonov"


# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 1
tmux send-keys "source /workspace/prune/setup.bash" Enter
#tmux send-keys "source .venv/bin/activate" Enter
#tmux send-keys "python ../pd_controller.py" Enter
tmux send-keys "uv run deployment/src/pd_controller.py"


# Attach to the tmux session
tmux -2 attach-session -t $session_name
