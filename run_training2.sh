#!/bin/bash

# Configuration
MAX_PARALLEL_JOBS=4

# Environment configurations as associative array
declare -A ENV_CONFIGS=(
    ["HalfCheetah-v5"]=""
    ["Hopper-v5"]=""
    ["Walker2d-v5"]=""
    ["Ant-v5"]=""
    ["Humanoid-v5"]=""
    ["Swimmer-v5"]="--start_timesteps 10000"
    #["InvertedPendulum-v4"]="--start_timesteps 1000"
    #["InvertedDoublePendulum-v4"]="--start_timesteps 1000"
    #["Reacher-v4"]="--start_timesteps 1000"
)

declare -g COMMANDS=()

# Function to generate commands
generate_commands() {
    local env=$1
    local num_runs=5  # Run each environment 10 times with different seeds
    
    for ((i=0; i<num_runs; i++)); do
        # Get additional parameters for specific environments
        local extra_params=${ENV_CONFIGS[$env]}
	    local env_seed=$(($i+5))
    	echo $env_seed	
        # Construct command with proper redirection
        COMMANDS+=("python main.py --policy GamidVae2 --env ${env} --seed ${i} ${extra_params} --file_append n3_1w5_lr_1e-3 > /dev/null 2>&1")
	#COMMANDS+=("python main.py --policy TD3 --env ${env} --seed ${i} ${extra_params}")
    done
}

# Create a temporary file to track running jobs
JOBS_FILE=$(mktemp)

# Initialize job counter
job_counter=0

# Function to wait for available slot
wait_for_slot() {
    while true; do
        # Count running jobs
        running_jobs=$(jobs -p | wc -l)
        echo "Current running jobs: $running_jobs"
        if [ $running_jobs -lt $MAX_PARALLEL_JOBS ]; then
            break
        fi
        sleep 1
    done
}

# Function to run a command
run_command() {
    local cmd=$1
    local job_id=$2
    
    echo "Starting job $job_id:"
    echo "$cmd"
    
    # Run the command in background
    eval "$cmd &"
    
    # Store job PID
    echo $! >> "$JOBS_FILE"
    
    echo "Launched job $job_id"
}

# Generate commands for each environment
for env_key in "${!ENV_CONFIGS[@]}"; do
    generate_commands "$env_key"
done

# Print total number of commands
echo "Total experiments to run: ${#COMMANDS[@]}"
echo "Will run up to $MAX_PARALLEL_JOBS jobs in parallel"
echo "Starting experiments..."
echo "----------------------------------------"

# Launch jobs
for cmd in "${COMMANDS[@]}"; do
    # Wait for available slot
    wait_for_slot
    
    # Increment job counter
    ((job_counter++))
    
    # Run command
    run_command "$cmd" "$job_counter"
    
    # Small sleep to prevent potential race conditions
    sleep 1
done

echo "----------------------------------------"
echo "Waiting for all jobs to complete..."

# Wait for all jobs to complete
while read -r pid; do
    if ps -p "$pid" > /dev/null; then
        wait "$pid" || { echo "Job $pid failed"; exit 1; }
    fi
done < "$JOBS_FILE"

# Cleanup
rm -f "$JOBS_FILE"

echo "All experiments completed successfully!"
