# isac-tradeoff

This repository contains the experimental code accompanying the paper ``Millimeter-Wave Gesture Recognition in ISAC: Does Reducing Sensing Airtime Hamper Accuracy?''. All code is provided as-is, with no guarantees whatsoever. To run the code, the dataset is required. Please refer to the paper for further information on the dataset.

This repository contains the following:
- `traintest.py`: The main script, which trains the classifier and evaluates its performance, providing performance as textual outputs, alongside a confusion matrix and saliency map as images.
- `convnet.py`: A helper file for the script above
- `plot_performances.py`: script to generate the graphs within the paper
- `runall.sh`: helper shell script to run multiple instances of traintest.py, with different parameters, in sequence
