import os
import sys
import csv
import argparse
import traceback
import os.path as osp

########################################################################
# Args Parsing


def validate_dataset_file(dataset_path):
    if not osp.isfile(dataset_path):
        return False
    if not dataset_path.lower().endswith('.csv'):
        return False
    return True


def positive_int(value):
    try:
        value_as_int = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: '{value}'. It must be an integer.")
    if value_as_int <= 0:
        raise argparse.ArgumentTypeError(f"Invalid value: '{value}'. It must be a strictly positive integer.")
    return value_as_int


# Args Parsing
########################################################################

class Neural_Network_Simulation:
    def __init__(self, args):
        if not validate_dataset_file(args.dataset):
            raise FileNotFoundError(f"Error : '{args.dataset}' is not correct or not set")
        
        self.dataset = []
        with open(args.dataset, 'r') as file:
            data_reader = csv.reader(file)
            for row in data_reader:
                if not row:
                    continue
                self.dataset.append(row)

        if args.learning_rate <= 0.0001:
            raise ValueError(f"Invalid value: learning_rate '{args.learning_rate}' is too small.")
        self.learning_rate = args.learning_rate

        self.early_stopping = args.early_stopping
        self.decay = args.decay
        self.plot = args.plot
        self.history = args.history

        self.hidden_layers = args.hidden_layers
        self.epochs = args.epochs
        self.batch_size = args.batch_size


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Train a Multilayer Perceptron network"
        )

        parser.add_argument(
            "dataset",
            type=str,
            help="Path to the dataset (CSV file)"
        )

        parser.add_argument(
            "--hidden_layers",
            nargs='+',
            type=positive_int,
            default=[24, 24, 24],
            help="Number of neurons per hidden layer"
        )

        parser.add_argument(
            "--epochs",
            type=positive_int,
            default=30,
            help="Numbers of epochs"
        )

        parser.add_argument(
            "--batch_size",
            type=positive_int,
            default=8,
            help="Batch size"
        )

        parser.add_argument(
            "--learning_rate",
            type=float,
            default=0.0314,
            help="Learning rate"
        )

        parser.add_argument(
            "--early_stopping",
            action="store_true",
            help="Stop an epoch prematurely to protect against overfitting"
        )

        parser.add_argument(
            "--decay",
            action="store_true",
            help="Learning rate decay with time"
        )

        parser.add_argument(
            "--plot",
            action="store_true",
            help="Display Loss/Accuracy learning curves at the end"
        )

        parser.add_argument(
            "--history",
            action="store_true",
            help="Print and plot the history of metrics obtained during training at the end"
        )

        args = parser.parse_args()

        simulation = Neural_Network_Simulation(args)
        print(simulation.dataset)

    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()