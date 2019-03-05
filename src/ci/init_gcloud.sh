#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

SDK_VERSION="236.0.0"

if [[ ! -z "${GCP_CLIENT_SECRET+x}" ]]; then
    echo "Downloading the Google Cloud SDK..."
    curl "https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-${SDK_VERSION}-linux-x86_64.tar.gz" > /tmp/gcloud-sdk.tar.gz

    echo "Extracting the Google Cloud SDK..."
    tar xzf /tmp/gcloud-sdk.tar.gz -C "${HOME}"

    echo "Logging in Google Cloud through the service account..."
    echo "${GCP_CLIENT_SECRET}" | base64 -d | gunzip > /dev/shm/gcp-client-secret.json
    ~/google-cloud-sdk/bin/gcloud auth activate-service-account --key-file /dev/shm/gcp-client-secret.json
    rm /dev/shm/gcp-client-secret.json
else
    echo "Skipping initializing the Google Cloud SDK since GCP_CLIENT_SECRET is missing"
fi
