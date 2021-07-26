import argparse

import clustering

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_name", type=str, help="pickle file which information is used for cluster analysis")
    parser.add_argument("training", type=str, help='yes if you want to train model, no if you want only prediction analysis')
    args = parser.parse_args()
    model = clustering.YieldClustering(args.file_name)
    if args.training == 'yes':
        model.train()
        print("It is trained")
        model.cluster_analysis()
        print("Analysis is done")
    else:
        model.cluster_analysis()
        print("Analysis is done")

