#./test.sh trained_model <input path to test.csv> output.csv (This will generate
#an output predictions file named output.csv in the present directory.)

python3 main.py --mode "test" --model_name "$1" --test_path "$2" --out_name "$3"

# ./test.sh "model" "./test.csv" "output.csv
