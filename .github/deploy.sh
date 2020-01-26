#! /bin/bash

set -ex

echo "Removing the current docs for master"
rm -rf out/master/ || exit 0

echo "Making the docs for master"
mkdir out/master/
cp util/gh-pages/index.html out/master
python ./util/export.py out/master/lints.json

if [[ -n $TAG_NAME ]]; then
  echo "Save the doc for the current tag ($TAG_NAME) and point current/ to it"
  cp -r out/master "out/$TAG_NAME"
  rm -f out/current
  ln -s "$TAG_NAME" out/current
fi

# Generate version index that is shown as root index page
cp util/gh-pages/versions.html out/index.html

cd out
cat <<-EOF | python - > versions.json
import os, json
print json.dumps([
    dir for dir in os.listdir(".") if not dir.startswith(".") and os.path.isdir(dir)
])
EOF

# Now let's go have some fun with the cloned repo
git config user.name "GHA CI"
git config user.email "gha@ci.invalid"

if git diff --exit-code --quiet; then
  echo "No changes to the output on this push; exiting."
  exit 0
fi

git add .
git commit -m "Automatic deploy to GitHub Pages: ${SHA}"

git push "$SSH_REPO" "$TARGET_BRANCH"
