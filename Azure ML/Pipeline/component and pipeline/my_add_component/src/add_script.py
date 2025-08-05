import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Add two numbers and save the result')
    parser.add_argument("--num1", type=int, help="The first number")
    parser.add_argument("--num2", type=int, help="The second number")
    parser.add_argument("--output_path", type=str, help="Path to save the result")
    args = parser.parse_args()
    
    add_result = args.num1 + args.num2
    
    output_file_path = os.path.join(args.output_path, "sum_result.txt")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    with open(output_file_path, "w") as f:
        f.write(str(add_result))

if __name__ == "__main__":
    main()