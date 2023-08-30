# ./train.sh <input path to train.csv> trained_model (This will create a model file
# named trained_model in the present directory.)

python3 main.py --mode "train" --train_path "$1" --model_name "$2"

# ./train.sh "./train.csv" "model"
