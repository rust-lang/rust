#!/bin/bash
# Check out all our submodules, but more quickly than using git by using one of
# our custom scripts

set -o errexit
set -o pipefail
set -o nounset

if [ ! -d ".git" ]; then
    echo "Error: This must run in the root of the repository"
    exit 1
fi

ci_dir=$(cd $(dirname $0) && pwd)/..
. "$ci_dir/shared.sh"

# On the beta channel we'll be automatically calculating the prerelease version
# via the git history, so unshallow our shallow clone from CI.
if [ "$(releaseChannel)" = "beta" ]; then
  git fetch origin --unshallow beta master
fi

function fetch_github_commit_archive {
    local module=$1
    local cached="download-${module//\//-}.tar.gz"
    retry sh -c "rm -f $cached && \
        curl -f -sSL -o $cached $2"
    mkdir $module
    touch "$module/.git"
    # On Windows, the default behavior is to emulate symlinks by copying
    # files. However, that ends up being order-dependent while extracting,
    # which can cause a failure if the symlink comes first. This env var
    # causes tar to use real symlinks instead, which are allowed to dangle.
    export MSYS=winsymlinks:nativestrict
    tar -C $module --strip-components=1 -xf $cached
    rm $cached
}

#included="src/llvm-project src/doc/book src/doc/rust-by-example"
included=""
modules="$(git config --file .gitmodules --get-regexp '\.path$' | cut -d' ' -f2)"
modules=($modules)
use_git=""
urls="$(git config --file .gitmodules --get-regexp '\.url$' | cut -d' ' -f2)"
urls=($urls)
# shellcheck disable=SC2068
for i in ${!modules[@]}; do
    module=${modules[$i]}
    if [[ " $included " = *" $module "* ]]; then
        commit="$(git ls-tree HEAD $module | awk '{print $3}')"
        git rm $module
        url=${urls[$i]}
        url=${url/\.git/}
        fetch_github_commit_archive $module "$url/archive/$commit.tar.gz" &
        bg_pids[${i}]=$!
        continue
    else
      # Submodule paths contained in SKIP_SUBMODULES (comma-separated list) will not be
      # checked out.
      if [ -z "${SKIP_SUBMODULES:-}" ] || [[ ! ",$SKIP_SUBMODULES," = *",$module,"* ]]; then
        use_git="$use_git $module"
      fi
    fi
done
retry sh -c "git submodule deinit -f $use_git && \
    git submodule sync && \
    git submodule update -j 16 --init --recursive --depth 1 $use_git"
#STATUS=0
#for pid in ${bg_pids[*]}
#do
#    wait $pid || STATUS=1
#done
#exit ${STATUS}
