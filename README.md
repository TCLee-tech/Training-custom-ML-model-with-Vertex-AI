# Training-custom-ML-model-with-Vertex-AI

Codelab: https://codelabs.developers.google.com/vertex-p2p-training#0  
Video: https://youtu.be/VRQXIiNLdAk?si=-NXP-PwD-VeYdljQ

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

Next, check if the currently active Google Cloud account configured in the Google Cloud SDK Shell is the correct one to use for authentication:
```
gcloud auth list
```
  - If you need to change to another Google Cloud account:
```
gcloud config set account 'ACCOUNT'
```
Also, check if the `project` property in the core section of the Cloud Shell is the correct one:
```
gcloud config list core/project
```
  - To alter the `project` property:
```
gcloud config set project 'PROJECT_ID'
```
When the settings for the SDK Shell are correct, log in to your Google Cloud account.
```
gcloud auth application-default login
```
Enable Compute Engine API, Artifact Registry API, Vertex AI API
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
<hr>

### 2. Access control with IAM
Use a service account to customize permissions available to a customs training container. Grant permissions for this service account to access Vertex AI and Cloud Storage resources.  
Refer to   
  - [Authentication to Vertex AI](https://cloud.google.com/vertex-ai/docs/authentication#on-gcp)  
  - [Vertex AI access control with IAM](https://cloud.google.com/vertex-ai/docs/general/access-control)

<hr>

### 3. Containerize training app
#### Step 1: Create Cloud Storage Bucket
Get ID of current Google Cloud project
```
gcloud config list --format 'value(core.project)'
```
Set `PROJECT_ID` environment variable
```
PROJECT_ID = <enter project ID here>
```
Create new bucket
```
BUCKET='gs://$(PROJECT_ID)-bucket'
gcloud storage buckets create $BUCKET --location=us-central1
```
#### Step 2: Copy training dataset to Cloud Storage bucket
In this exercise, a sample dataset of flower images is used. The tar file is downloaded and untar.
```
wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
tar xvzf flower_photos.tgz
```
Copy dataset to Cloud Storage bucket
```
gcloud storage cp --recursive flower_photos $BUCKET
```
Reference:   
  - [Discover object storage with the gcloud tool](https://cloud.google.com/storage/docs/discover-object-storage-gcloud)
  - [gcloud storage cp](https://cloud.google.com/sdk/gcloud/reference/storage/cp)






 
