#!/bin/bash

set -ex

echo "Removing the current docs for master"
rm -rf out/master/ || exit 0

echo "Making the docs for master"
mkdir out/master/
cp util/gh-pages/index.html out/master
cp util/gh-pages/lints.json out/master

if [[ -n $TAG_NAME ]]; then
  echo "Save the doc for the current tag ($TAG_NAME) and point stable/ to it"
  cp -Tr out/master "out/$TAG_NAME"
  rm -f out/stable
  ln -s "$TAG_NAME" out/stable
fi

if [[ $BETA = "true" ]]; then
  echo "Update documentation for the beta release"
  cp -r out/master/* out/beta
fi

# Generate version index that is shown as root index page
cp util/gh-pages/versions.html out/index.html

echo "Making the versions.json file"
python3 ./util/versions.py out

# Now let's go have some fun with the cloned repo
cd out
git config user.name "GHA CI"
git config user.email "gha@ci.invalid"

if [[ -n $TAG_NAME ]]; then
  # track files, so that the following check works
  git add --intent-to-add "$TAG_NAME"
  if git diff --exit-code --quiet -- $TAG_NAME/; then
    echo "No changes to the output on this push; exiting."
    exit 0
  fi
  # Add the new dir
  git add "$TAG_NAME"
  # Update the symlink
  git add stable
  # Update versions file
  git add versions.json
  git commit -m "Add documentation for ${TAG_NAME} release: ${SHA}"
elif [[ $BETA = "true" ]]; then
  if git diff --exit-code --quiet -- beta/; then
    echo "No changes to the output on this push; exiting."
    exit 0
  fi
  git add beta
  git commit -m "Automatic deploy to GitHub Pages (beta): ${SHA}"
else
  if git diff --exit-code --quiet; then
    echo "No changes to the output on this push; exiting."
    exit 0
  fi
  git add .
  git commit -m "Automatic deploy to GitHub Pages: ${SHA}"
fi

git push "$SSH_REPO" "$TARGET_BRANCH"
