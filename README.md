Name:
Classification of GTZAN dataset

Description:
This project can be used to:
1. Extract data-type features from a single music file. Features extracted include: Chromatogram CQT, STFT, CENS, Root Mean Square Energy,
   Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, zero crossing rate, mel-frequency cepstral coefficients, etc.
2. Extract the above features of a batch of music files.
3. Edit the extracted feature file and store it in a CSV file.
4. Users can use their own feature data files or the feature data files existing in the project to classify GTZAN data sets.
5. This project uses fully connected DNN as the classification tool, and users can customize the form of DNN, that is, whether to call 
   the dropout layer to alleviate the overfitting of DNN.All the neural networks in this project have been debugged and can be used directly,
   and they perform well on the GTZAN data set.
   
Installation:
In order to use this project, you first need to download the project locally and then install the required dependencies by: Open the command line, 
enter the project path and type: pip install-r requirements.txt
(if 'pip' is not installed in your local PC, please call the above command after installation of 'pip', details please refer to: https://pip.pypa.io/en/stable/reference/pip_install/).

Usage:
You need to use this project in the command line.
First you need to go to the path of the current project on the command line.
Then use python main.py <Parameters>
Customize your own project parameters by giving the following values on the command line:
--id： Customize the <ID> of this round of training and testing: the results of training and testing will be saved in the folder named <ID>. The script will
	   create a folder named ‘results’ in the directory where the project is running, and this folder will include a user-defined folder named <ID>.
--generate-features： Customize running mode: training&test mode and feature extraction mode. By default, the script goes into train&test mode. If the user
       feeds path to the --generate-features on the command line, the script will extract the features of the audio from that path.
--load-features-path： Load the feature file. By default, the script automatically looks for the feature file from the path where the current project is running.
       If the user has other feature files stored elsewhere on the PC, you can load the feature files by calling --load-features-path.
--epochs： The number of epochs of training, the default value is 100.
--seed： Random seed of shuffling data set, the default value is 0.
--batch-size： Batch size. Default value is 64.
--validation_split： The proposion of the validation. The default value is 0.2.
--test-size： The proposion of the test set. The default value is 0.2.
--disable-dropout: Customize whether to call Dropout or not.
--test: Customize whether the current script is only for testing.
--models: Call the previously trained model file, skip the training process and directly enter the testing process. If the user enters the corresponding path 
           in the --models, the script will go to that path to load the model file. If the user does not enter a path, the script will look for the model file
		   in the folder where the current project is running.
		   
Result:
Using the neural network in this project to classify the GTZAN data set, the training time is very short, and the CPU can complete the convergence within ten 
seconds without the acceleration of GPU.
Using the neural network without Dropout processing, the model will overfit the training data and perform well in the training set, but not in the test set, 
with a test accuracy of about 66%. However, after using the neural network processed by Dropout, the model has been greatly improved, the phenomenon of overfitting
has been significantly alleviated, and the performance in the test set has been increased from 66% at the beginning to 75.5%.

Support:
If you have any questions or problems of using this project, please contact qiang.wang@ucdconnect.ie

Contributing:
This project can be used to directly classify the GTZAN dataset or similar datasets. Feature extraction script in the project can be directly used to extract the 
features of various music files. The application method is very simple. Users can take part of the script in the project according to their own needs, and then 
implement them in their own project.

Authors and acknowledgment：
Authors： Qiang Wang
Acknowledgment: Thanks to Professor Guenole Silvestre for the course in this semester and the hard work of each TA, I have gained a lot. Thank you very much.