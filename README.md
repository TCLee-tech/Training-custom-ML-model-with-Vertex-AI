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






 
