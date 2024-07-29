#!/bin/bash

# List of predefined commands
commands=(
    ##"dora run -f 0f86b184"
    ##"dora run -f 2d9ba778"
    "dora run -f 2d92d2fd"
    ##"dora run -f 27be639a"
    ##"dora run -f 81b48f2b"
    ##"dora run -f b742cdd5"
    ##"dora run -f fb217f1a"
    "dora run -f 7ccb0b5c"
    "dora run -f c649697b"
)

# Execute each command sequentially
for cmd in "${commands[@]}"; do
    echo "Executing: $cmd"
    eval $cmd
    if [ $? -ne 0 ]; then
        echo "Command failed: $cmd"
        exit 1
    fi
done

echo "All commands executed successfully!"
