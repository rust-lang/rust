#!/bin/bash

# Automatically deploy on gh-pages

set -ex

SOURCE_BRANCH="master"
TARGET_BRANCH="gh-pages"

# Save some useful information
REPO=$(git config remote.origin.url)
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
SHA=$(git rev-parse --verify HEAD)

# Clone the existing gh-pages for this repo into out/
(
    git clone "$REPO" out
    cd out
    git checkout $TARGET_BRANCH
)

echo "Removing the current docs for master"
rm -rf out/master/ || exit 0

echo "Making the docs for master"
mkdir out/master/
cp util/gh-pages/index.html out/master
python ./util/export.py out/master/lints.json

if [ -n "$TRAVIS_TAG" ]; then
    echo "Save the doc for the current tag ($TRAVIS_TAG) and point current/ to it"
    cp -r out/master "out/$TRAVIS_TAG"
    rm -f out/current
    ln -s "$TRAVIS_TAG" out/current
fi

# Generate version index that is shown as root index page
(
    cp util/gh-pages/versions.html out/index.html

    cd out
    python -c '\
        import os, json;\
        print json.dumps([\
            dir for dir in os.listdir(".")\
            if not dir.startswith(".") and os.path.isdir(dir)\
        ])' > versions.json
)

# Pull requests and commits to other branches shouldn't try to deploy, just build to verify
if [ "$TRAVIS_PULL_REQUEST" != "false" ] || [ "$TRAVIS_BRANCH" != "$SOURCE_BRANCH" ]; then
    # Tags should deploy
    if [ -z "$TRAVIS_TAG" ]; then
        echo "Generated, won't push"
        exit 0
    fi
fi

# Get the deploy key by using Travis's stored variables to decrypt deploy_key.enc
ENCRYPTION_LABEL=e3a2d77100be
ENCRYPTED_KEY_VAR="encrypted_${ENCRYPTION_LABEL}_key"
ENCRYPTED_IV_VAR="encrypted_${ENCRYPTION_LABEL}_iv"
ENCRYPTED_KEY=${!ENCRYPTED_KEY_VAR}
ENCRYPTED_IV=${!ENCRYPTED_IV_VAR}
openssl aes-256-cbc -K "$ENCRYPTED_KEY" -iv "$ENCRYPTED_IV" -in .github/deploy_key.enc -out .github/deploy_key -d
chmod 600 .github/deploy_key
eval $(ssh-agent -s)
ssh-add .github/deploy_key

# Now let's go have some fun with the cloned repo
cd out
git config user.name "Travis CI"
git config user.email "travis@ci.invalid"

if [ -z "$(git diff --exit-code)" ]; then
    echo "No changes to the output on this push; exiting."
    exit 0
fi

git add .
git commit -m "Automatic deploy to GitHub Pages: ${SHA}"

# Now that we're all set up, we can push.
git push "$SSH_REPO" "$TARGET_BRANCH"
