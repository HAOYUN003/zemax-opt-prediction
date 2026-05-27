# zemax-opt-prediction
Zemax optical aberration prediction system based on Random Forest. Predict RMS spot radius (optical aberration) of lens system by inputting optical parameters including R1, R2, lens thickness and image plane position. Includes model training script and real‑time prediction module.

1. auto.py
The integrated main running program of the project. It automatically completes full-process tasks: reading CCD spot images, invoking RMS calculation module, inputting computed optical parameters into machine learning model, and outputting aberration prediction results.

2. calculate.py
Core optical computation module. It contains complete pixel calculation codes, computes the centroid position and RMS radius of laser spot from grayscale CCD images, and exports standardized optical feature data for subsequent modeling.

3. kaggle.xlsx
Experimental dataset file in Excel format. It stores batches of experimental measured RMS values and corresponding optical aberration labels, which serves as training and testing dataset for building the machine learning model.

4. opt_model.pkl
Pickle-formatted file of trained machine learning model. After training on experimental optical data, the regression model is saved here, responsible for predicting optical aberration conditions using spot RMS parameters.

5. opt_scaler.pkl
Data normalization parameter file. It preserves the scaling parameters of feature standardization during model training. It ensures consistent data dimension and distribution for new experimental data before model prediction, avoiding model prediction deviation.

6. predict.py
Independent prediction module. Separately calls the saved model and scaler file, receives calculated RMS data, and independently completes optical aberration prediction for single or multiple groups of experimental data.

7. run.py
Simplified entry startup file. It can directly call core functions of auto.py to execute the whole project with one click, reducing tedious operation steps for experimental data batch processing.
