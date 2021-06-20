# Note to people running shellcheck: this file should only be sourced, not executed directly.

set -e

export LD_LIBRARY_PATH="$(rustc --print sysroot)/lib"
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
