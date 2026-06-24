import os
import argparse
import traceback

class Neural_Network_Simulation:
    def __init__(self, args):
        self.dataset_file = args.dataset
        self.hidden_layers = args.hidden_layers
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.early_stopping = args.early_stopping
        self.decay = args.decay
        self.plot = args.plot
        self.history = args.history

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
            type=int,
            default=[24, 24, 24],
            help="Number of neurons per hidden layer"
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=30,
            help="Numbers of epochs"
        )

        parser.add_argument(
            "--batch_size",
            type=int,
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


        
        print(f"Dataset chargé : {args.dataset}")
        print(f"Couches cachées : {args.hidden_layers}")

    except Exception as e:
        print(f"An exception has been caught: {type(e).__name__} - {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()