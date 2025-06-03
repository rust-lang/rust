#!/bin/bash
# A quick smoke test to make sure publish_toolstate.py works.

set -euo pipefail
IFS=$'\n\t'

rm -rf rust-toolstate
git clone --depth=1 https://github.com/rust-lang-nursery/rust-toolstate.git
cd rust-toolstate
python3 "../../src/tools/publish_toolstate.py" "$(git rev-parse HEAD)" \
    "$(git log --format=%s -n1 HEAD)" "" ""
cd ..
rm -rf rust-toolstate
