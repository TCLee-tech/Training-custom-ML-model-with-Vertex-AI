# Training-custom-ML-model-with-Vertex-AI

Please refer to the linked [Codelab](https://codelabs.developers.google.com/vertex-p2p-training#0) and [Video](https://youtu.be/VRQXIiNLdAk?si=-NXP-PwD-VeYdljQ) for an introduction.  
  
### 1. Enable APIs
To use Google Cloud Command Line Interface(CLI) on your local computer to manage Google Cloud resources and services via command line instructions and scripts, install the Google Cloud CLI.   
  - Installation instructions are [here](https://cloud.google.com/sdk/docs/install).  
  - Or, to update existing SDK to the latest version, run  
```
gcloud components update
```
  - To view the components available for installation or removal, use
```
gcloud components list
```
![gcloud components update](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/e1dec8eb5b950f433c3b799e9e9ab19cb0b7cf67/gcloud%20cli%20components%20update.jpg)

Next, initialize, configure and authorize the gcloud CLI:
```
gcloud init
```
If necessary, enable Compute Engine API, Artifact Registry API, Vertex AI API
```
gcloud services enable \
compute.googleapis.com \
artifactregistry.googleapis.com \
aiplatform.googleapis.com
```
To verify, list the services:
```
gcloud services list
```
Reference: [gcloud CLI cheat sheet](https://cloud.google.com/sdk/docs/cheatsheet)  

<hr>

### 2. Connect to Cloud Shell   
Launch a Cloud Shell session. Establish an interactive SSH connection from the local gcloud CLI.  
```
gcloud cloud-shell ssh --authorize-session
```
Reference: [Use Cloud Shell with gcloud CLI](https://cloud.google.com/shell/docs/using-cloud-shell-with-gcloud-cli)  

<hr>

### 3. Access control with IAM
Use a service account to customize permissions available to a customs training container. Grant permissions for this service account to access Vertex AI and Cloud Storage resources.  
Refer to   
  - [Authentication to Vertex AI](https://cloud.google.com/vertex-ai/docs/authentication#on-gcp)  
  - [Vertex AI access control with IAM](https://cloud.google.com/vertex-ai/docs/general/access-control)

<hr>

### 4. Containerize training app  
In the Cloud Shell session window launched:

#### Step 1: Create Cloud Storage Bucket
Set `PROJECT_ID` environment variable
```
export PROJECT_ID =$(gcloud config get-value project)
```
Create new bucket
```
BUCKET="gs://${PROJECT_ID}-bucket"
gcloud storage buckets create $BUCKET --location=us-central1
```
  - note 1: When setting environment variables, must use " ..." and not ' ... '
  - note 2: `${}` curly braces are for field replacement. $( ) is for command.   

To verify,
```
echo $PROJECT_ID
echo $BUCKET
```

#### Step 2: Copy training dataset to Cloud Storage bucket
In this exercise, a sample dataset of flower images is used. First, we download the .tgz source distribution file and untar it.
```
wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar xvzf flower_photos.tgz
```
Then, copy the dataset from Cloud Shell to a Cloud Storage bucket
```
gcloud storage cp --recursive flower_photos $BUCKET
```
Reference:   
  - [Discover object storage with the gcloud tool](https://cloud.google.com/storage/docs/discover-object-storage-gcloud)
  - [gcloud storage cp](https://cloud.google.com/sdk/gcloud/reference/storage/cp)

To upload a dataset from your local computer to Google Cloud Storage, use the syntax `gcloud storage cp [source url] gs://[GCS bucket name]`. 
  - use the full [source url path] from the "Properties" option of the right-click menu for the file/folder.
  - if upload is successful, an output "Completed files number of files/total number of files | xxx/xxxB" would be printed in the terminal.   
  - succeessful upload can also be verified from the Google Cloud Console > Cloud Storage > [Bucket] content.  
  - either the gcloud CLI or Cloud Shell can be used to upload a local dataset.
  - Storage Object User role is needed for the upload and _storage.buckets.list_ permission is needed for upload using Google Cloud console. Refer to [Upload objects from a file system](https://cloud.google.com/storage/docs/uploading-objects)

![local storage to GCS](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/e1dec8eb5b950f433c3b799e9e9ab19cb0b7cf67/local%20computer%20to%20GCS.jpg)

#### Step 3: Write training code
Create new directory called `flowers` and change to this directory:
```
mkdir flowers
cd flowers
```
Create a `trainer` sub-directory and a `task.py` Python file where you'll add the code for training the custom model:
```
mkdir trainer
touch trainer/task.py
```
Open `task.py` and paste the model training code below.  
Replace {your-gcs-bucket} with the name of your Cloud Storage bucket.  
```
cd trainer
nano task.py
```
Custom mode training code:
```
import tensorflow as tf
import numpy as np
import os

## Replace {your-gcs-bucket} !!
BUCKET_ROOT='/gcs/{your-gcs-bucket}'

# Define variables
NUM_CLASSES = 5
EPOCHS=10
BATCH_SIZE = 32

IMG_HEIGHT = 180
IMG_WIDTH = 180

DATA_DIR = f'{BUCKET_ROOT}/flower_photos'

def create_datasets(data_dir, batch_size):
  '''Creates train and validation datasets.'''
  
  train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size)

  validation_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size)

  train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
  validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

  return train_dataset, validation_dataset


def create_model():
  '''Creates model.'''

  model = tf.keras.Sequential([
    tf.keras.layers.Resizing(IMG_HEIGHT, IMG_WIDTH),
    tf.keras.layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
  ])
  return model

# CREATE DATASETS
train_dataset, validation_dataset = create_datasets(DATA_DIR, BATCH_SIZE)

# CREATE/COMPILE MODEL
model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# TRAIN MODEL
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=EPOCHS
)

# SAVE MODEL
model.save(f'{BUCKET_ROOT}/model_output')
```   

#### Step 4: Create Dockerfile  
Create a Dockerfile at the same level as the training code folder:
```
cd ..
touch Dockerfile
```
Add commands to Dockerfile:
```
FROM gcr.io/deeplearning-platform-release/tf2-gpu.2-8

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
```
Reference:  
[Create a custom container image for training](https://cloud.google.com/vertex-ai/docs/training/create-custom-container#create_a_dockerfile)   
[List of Deep Learning containers](https://cloud.google.com/deep-learning-containers/docs/choosing-container#choose_a_container_image_type?utm_campaign=CDR_sar_aiml_ucaiplabs_011321&utm_source=external&utm_medium=web)  
[How to Package Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)  

#### Build container and push to Artifact Registry  
In the Cloud Shell (not gcloud CLI), obtain credentials for a Docker repository in us-central1 region:
```
gcloud auth configure-docker \
us-central1-docker.pkg.dev
```
Then, create a Docker repo in Artifact Registry:
```
REPO_NAME='flower-app'
gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=us-central1 --description="Docker repository"
```
Define a IMAGE_URL variable for use with Docker commands:
```
IMAGE_URI=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/flower_image:latest
```
Build Docker container








 
