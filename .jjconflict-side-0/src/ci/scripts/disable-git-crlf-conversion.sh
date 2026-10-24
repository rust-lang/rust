#!/bin/bash
# Disable automatic line ending conversion, which is enabled by default on
# GitHub's Windows image. Having the conversion enabled caused regressions both
# in our test suite (it broke miri tests) and in the ecosystem, since we
# started shipping install scripts with CRLF endings instead of the old LF.
#
# Note that we do this a couple times during the build as the PATH and current
# user/directory change, e.g. when mingw is enabled.

set -euo pipefail
IFS=$'\n\t'

git config --replace-all --global core.autocrlf false
