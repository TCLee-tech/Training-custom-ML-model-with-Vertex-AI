# Training-custom-ML-model-with-Vertex-AI
Workflow for using a custom dataset to train a custom ML model.
  - a custom dataset for training a ML model can be curated on your own computer before uploading to the cloud.
  - an image dataset of flowers is used as an example in this repo
  - the ML model groups images into categories
  - at least 1,000 training samples per category is recommended
  - clicking through cells in a Jupyter notebook is fine for initial experimentation. This workflow allows the use of a custom Python training code and a custom ML container.
  - model endpoint can serve predictions
  - can scale up, to zero, or torn down as needed.
  - adaptable for scripting

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
export PROJECT_ID=$(gcloud config get-value project)
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
  - for a Windows desktop, use the full [source url path] from the "Properties" option of the right-click menu for the file/folder.
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
Custom model training code:
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
To convert custom model codes in a Jupyter notebook into a Python file, use `jupyter nbconvert task.ipynb --to python`. Will get `task.py`

#### Step 4: Create Dockerfile  
Create a Dockerfile at the same level as the training code folder:
```
cd ..
touch Dockerfile
```
![Dockerfile](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/8596197c494f6aec6046fa56eb051067ba7c9094/Dockerfile.jpg)  

Add commands to Dockerfile:
```
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-15.py310

WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.task"]
```
Note: the base image in the above Dockerfile is a more current version than the one in the Google codelab.  

Reference:  
[Create a custom container image for training](https://cloud.google.com/vertex-ai/docs/training/create-custom-container#create_a_dockerfile)   
[List of Deep Learning containers](https://cloud.google.com/deep-learning-containers/docs/choosing-container#choose_a_container_image_type?utm_campaign=CDR_sar_aiml_ucaiplabs_011321&utm_source=external&utm_medium=web)  
[How to Package Python Projects](https://packaging.python.org/en/latest/tutorials/packaging-projects/)  

#### Step 5: Build container and push to Artifact Registry  
In the Cloud Shell (not gcloud CLI), obtain credentials for a Docker repository in the _us-central1_ region:
```
gcloud auth configure-docker \
us-central1-docker.pkg.dev
```
Set three environment variables (PROJECT_ID, REPO_NAME, IMAGE_URL) for use when creating the Docker repo and running Docker commands:
```
PROJECT_ID=$(gcloud config list --format 'value(core.project)')   
REPO_NAME='flower-app'  
IMAGE_URL=us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/flower_image:latest
```
To verify,
```
echo $PROJECT_ID
echo $REPO_NAME
echo $IMAGE_URL
```
Then, create a Docker repo in Artifact Registry:
```
gcloud artifacts repositories create $REPO_NAME --repository-format=docker --location=us-central1 --description="Docker repository"
```
Build the Docker container:
```
docker build ./ -t $IMAGE_URL
```
Push built Docker container to Artifact Registry:
```
docker push $IMAGE_URL
```
Note: Cloud Shell is a 5GB VM. The TensorFlow container base image is about 2.5GB. If Cloud Shell returns an error message about insufficient space, you can delete the _flower_photos.tgz_ compressed file and the _flower_photos_ folder. The flowers dataset was uploaded to Google Cloud Storage bucket, and that will be the copy used for model training.
```
rm -r flower_photos
delete flower_photos.tgz
```
To verify, in Google Cloud console > Artifact Registry, select _flower_app_ repository to find _flower_image_ Docker container image.
![Docker image in Artifact Registry](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/8596197c494f6aec6046fa56eb051067ba7c9094/docker%20image%20in%20artifact%20registry.jpg)

<hr>

### 5. Run a custom training job on Vertex AI

To run the custom training with GPU (e.g. NVIDIA_TESLA_V100),
```
gcloud ai custom-jobs create \
--display-name='flower-sdk-job' \
--region=us-central1 \
--worker-pool-spec=replica-count=1,machine-type='n1-standard-8',accelerator-type='NVIDIA_TESLA_V100',accelerator-count=1,container-image-uri=$IMAGE_URL
```
Refer to [gcloud ai custom-jobs create command](https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create) for values to key flags.  

![gcloud ai custom-jobs create command](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/7992f815f94f241c6672772bdb00904cd7ba1efb/gcloud%20ai%20custom-jobs%20create%20command.jpg)
 
If the following error message is encountered:
> ERROR: (gcloud.ai.custom-jobs.create) RESOURCE_EXHAUSTED: The following quota metrics exceed quota limits: aiplatform.googleapis.com/custom_model_training_nvidia_v100_gpus   

you can also run the custom ML training without GPU:
```
gcloud ai custom-jobs create \
--display-name='flower-sdk-job' \
--region=us-central1 \
--worker-pool-spec=replica-count=1,machine-type='n1-standard-8',container-image-uri=$IMAGE_URL
```
To verify and view progress of your custom training job, use the following commands from the Cloud Shell
```
gcloud ai custom-jobs describe projects/xxxx/locations/us-central1/customJobs/xxx
```
or   
```
gcloud ai custom-jobs stream-logs projects/xxxx/locations/us-central1/customJobs/xxx
```
References:

[gcloud ai custom-jobs describe command](https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/describe)   
[gcloud ai custom-jobs stream-logs command](https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/stream-logs)  


In Google Cloud console, navigate to Vertex AI > "Training" under MODEL DEVELOPMENT > "CUSTOM JOBS" tab to verify status of training:
![custom training job](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/7992f815f94f241c6672772bdb00904cd7ba1efb/custom%20training%20job.jpg)   

On successful completion of custom ML training:  

![custom training job completed](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/7992f815f94f241c6672772bdb00904cd7ba1efb/custom%20training%20job%20completed.jpg)  

The trained model is saved to Google Cloud Storage, as written in Python training codes:  

![trained model in GCS](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/03447a946dd446aa544e548fc85cb6c270ea4cdc/trained%20model%20in%20GCS.jpg)

<hr>

:rainbow::rainbow::rainbow::rainbow::rainbow::fountain::fountain::fountain::rainbow::rainbow::rainbow::rainbow::rainbow:

# Getting predictions from custom trained model

Instructions below use the trained model artifact in Google Cloud Storage, from the steps above.

Please refer to linked [video](https://youtu.be/-9fU1xwBQYU?si=7BY-2FLRO987YB_9) and [codelab](https://codelabs.developers.google.com/vertex-p2p-predictions#0)

<hr>

### 1. Upload model to Vertex AI's Model Registry
The trained model artifact is in a folder `/model_output` in Google Cloud Storage. It needs to be packaged with a pre-built container that supports the TensorFlow runtime (note: a custom container stored in Artifact Registry can be used too). The packaged container image needs to be uploaded to Vertex AI's Model Registry.
```
gcloud ai models upload \
--display-name=flowers \
--container-image-uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-9:latest" \
--artifact-uri=${BUCKET}/model_output \
--region=us-central1
```
References:   
[gcloud ai models upload](https://cloud.google.com/sdk/gcloud/reference/ai/models/upload)  
[list of pre-built container images for Vertex AI prediction](https://console.cloud.google.com/artifacts/docker/vertex-ai/us/prediction)  
  - supported runtimes: TensorFlow, scikit-learn, pytorch

To verify,
```
gcloud ai models list --region=us-central1
```
Reference:
[gcloud ai models list](https://cloud.google.com/sdk/gcloud/reference/ai/models/list)

To verify in Google Cloud console, navigate to Vertex AI > Model Registry under MODEL DEVELOPMENT:  

![model uploaded to Vertex AI Model Registry](https://github.com/TCLee-tech/Training-custom-ML-model-with-Vertex-AI/blob/7c2e6ad1564421505c394e2a00609ba67fac1f73/model%20uploaded%20to%20Model%20Registry.jpg)
