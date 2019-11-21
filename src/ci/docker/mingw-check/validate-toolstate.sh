#!/bin/bash
# A quick smoke test to make sure publish_tooolstate.py works.

set -euo pipefail
IFS=$'\n\t'

rm -rf rust-toolstate
git clone --depth=1 https://github.com/rust-lang-nursery/rust-toolstate.git
cd rust-toolstate
python3 "../../src/tools/publish_toolstate.py" "$(git rev-parse HEAD)" \
    "$(git log --format=%s -n1 HEAD)" "" ""
# Only check maintainers if this build is supposed to publish toolstate.
# Builds that are not supposed to publish don't have the access token.
if [ -n "${TOOLSTATE_PUBLISH+is_set}" ]; then
  TOOLSTATE_VALIDATE_MAINTAINERS_REPO=rust-lang/rust python3 \
      "../../src/tools/publish_toolstate.py"
fi
cd ..
rm -rf rust-toolstate
