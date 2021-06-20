# Note to people running shellcheck: this file should only be sourced, not executed directly.

set -e

dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../build"; pwd)
export LD_LIBRARY_PATH="$(rustc --print sysroot)/lib:"$dir"/lib"
export DYLD_LIBRARY_PATH=$LD_LIBRARY_PATH
