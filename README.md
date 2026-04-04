# Gaia: Semantic Scene Segmentation

Gaia is an end-to-end semantic segmentation pipeline for off-road desert environments, developed for Duality AI’s Offroad Semantic Scene Segmentation challenge.

## Project Structure
```
Gaia/
├── config.yaml          # Hyperparameters and path setups
├── requirements.txt     # Dependency list
├── README.md            # Setup and instructions
├── report_template.md   # Final report wrapper
├── src/                 # Reusable modules
│   ├── dataset.py       # Augmentations and Loading / Remapping
│   ├── model.py         # DeepLabV3+ Initialization
│   ├── metrics.py       # Loss and IoU Calculation
│   └── utils.py         # Visualizations
├── train.py             # Training & Validation Entrypoint
└── test.py              # Inference and metric visualization on unseen data
```

## Environment Setup
1. Create a Python environment (e.g. `conda create -n antigravity python=3.10`).
2. Run: `pip install -r requirements.txt`.

## How to Train
Run the training script pointing to the config:
```bash
python train.py --config config.yaml
```
*   This will leverage the CE+Dice Loss configuration.
*   It logs models that maximize the validation Mean IoU into `saved_model_weights/best.pth`.

## How to Test
Evaluate your trained best model automatically via the test script:
```bash
python test.py --config config.yaml --weights saved_model_weights/best.pth
```
*   Calculates overall mIoU against the unseen environment.
*   Outputs standard visualizations into `runs/visualizations/`.
*   Outputs lowest IoU scenes automatically into `runs/failure_cases/` for reporting.

## Expected Output Format
After testing, the `runs/` folder will be populated as follows:
* `runs/test/` - Contains the raw predicted `.png` masks mapped properly.
* `runs/visualizations/` - Overlay of predictions vs ground truth logic if test images have seg labels.
* `runs/failure_cases/` - Contains mispredictions comparing specific regions against reality for analysis.
