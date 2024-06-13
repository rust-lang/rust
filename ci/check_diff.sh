#!/bin/bash

set -e

function print_usage() {
    echo "usage check_diff REMOTE_REPO FEATURE_BRANCH [COMMIT_HASH] [OPTIONAL_RUSTFMT_CONFIGS]"
}

if [ $# -le 1 ]; then
    print_usage
    exit 1
fi

REMOTE_REPO=$1
FEATURE_BRANCH=$2
OPTIONAL_COMMIT_HASH=$3
OPTIONAL_RUSTFMT_CONFIGS=$4

# OUTPUT array used to collect all the status of running diffs on various repos
STATUSES=()

# Clone a git repository and cd into it.
#
# Parameters:
# $1: git clone url
# $2: directory where the repo should be cloned
function clone_repo() {
    GIT_TERMINAL_PROMPT=0 git clone --quiet $1 --depth 1 $2 && cd $2
}

# Initialize Git submodules for the repo.
#
# Parameters
# $1: list of directories to initialize
function init_submodules() {
    git submodule update --init $1
}

# Run rusfmt with the --check flag to see if a diff is produced.
#
# Parameters:
# $1: Path to a rustfmt binary
# $2: Output file path for the diff
# $3: Any additional configuration options to pass to rustfmt
#
# Globals:
# $OPTIONAL_RUSTFMT_CONFIGS: Optional configs passed to the script from $4
function create_diff() {
    local config;
    if [ -z "$3" ]; then
        config="--config=error_on_line_overflow=false,error_on_unformatted=false"
    else
        config="--config=error_on_line_overflow=false,error_on_unformatted=false,$OPTIONAL_RUSTFMT_CONFIGS"
    fi

    for i in `find . | grep "\.rs$"`
    do
        $1 --unstable-features --skip-children --check --color=always $config $i >> $2 2>/dev/null
    done
}

# Run the master rustfmt binary and the feature branch binary in the current directory and compare the diffs
#
# Parameters
# $1: Name of the repository (used for logging)
#
# Globals:
# $RUSFMT_BIN: Path to the rustfmt master binary. Created when running `compile_rustfmt`
# $FEATURE_BIN: Path to the rustfmt feature binary. Created when running `compile_rustfmt`
# $OPTIONAL_RUSTFMT_CONFIGS: Optional configs passed to the script from $4
function check_diff() {
    echo "running rustfmt (master) on $1"
    create_diff $RUSFMT_BIN rustfmt_diff.txt

    echo "running rustfmt (feature) on $1"
    create_diff $FEATURE_BIN feature_diff.txt $OPTIONAL_RUSTFMT_CONFIGS

    echo "checking diff"
    local diff;
    # we don't add color to the diff since we added color when running rustfmt --check.
    # tail -n + 6 removes the git diff header info
    # cut -c 2- removes the leading diff characters("+","-"," ") from running git diff.
    # Again, the diff output we care about was already added when we ran rustfmt --check
    diff=$(
        git --no-pager diff --color=never \
        --unified=0 --no-index rustfmt_diff.txt feature_diff.txt 2>&1 | tail -n +6 | cut -c 2-
    )

    if [ -z "$diff" ]; then
        echo "no diff detected between rustfmt and the feature branch"
        return 0
    else
        echo "$diff"
        return 1
    fi
}

# Compiles and produces two rustfmt binaries.
# One for the current master, and another for the feature branch
#
# Parameters:
# $1: Directory where rustfmt will be cloned
#
# Globals:
# $REMOTE_REPO: Clone URL to the rustfmt fork that we want to test
# $FEATURE_BRANCH: Name of the feature branch
# $OPTIONAL_COMMIT_HASH: Optional commit hash that will be checked out if provided
function compile_rustfmt() {
    RUSTFMT_REPO="https://github.com/rust-lang/rustfmt.git"
    clone_repo $RUSTFMT_REPO $1
    git remote add feature $REMOTE_REPO
    git fetch feature $FEATURE_BRANCH

    CARGO_VERSION=$(cargo --version)
    echo -e "\ncompiling with $CARGO_VERSION\n"

    # Because we're building standalone binaries we need to set `LD_LIBRARY_PATH` so each
    # binary can find it's runtime dependencies. See https://github.com/rust-lang/rustfmt/issues/5675
    # This will prepend the `LD_LIBRARY_PATH` for the master rustfmt binary
    export LD_LIBRARY_PATH=$(rustc --print sysroot)/lib:$LD_LIBRARY_PATH

    echo "Building rustfmt from src"
    cargo build -q --release --bin rustfmt && cp target/release/rustfmt $1/rustfmt

    if [ -z "$OPTIONAL_COMMIT_HASH" ] || [ "$FEATURE_BRANCH" = "$OPTIONAL_COMMIT_HASH" ]; then
        git switch $FEATURE_BRANCH
    else
        git switch $OPTIONAL_COMMIT_HASH --detach
    fi

    # This will prepend the `LD_LIBRARY_PATH` for the feature branch rustfmt binary.
    # In most cases the `LD_LIBRARY_PATH` should be the same for both rustfmt binaries that we build
    # in `compile_rustfmt`, however, there are scenarios where each binary has different runtime
    # dependencies. For example, during subtree syncs we bump the nightly toolchain required to build
    # rustfmt, and therefore the feature branch relies on a newer set of runtime dependencies.
    export LD_LIBRARY_PATH=$(rustc --print sysroot)/lib:$LD_LIBRARY_PATH

    echo "Building feature rustfmt from src"
    cargo build -q --release --bin rustfmt && cp target/release/rustfmt $1/feature_rustfmt

    echo -e "\nRuntime dependencies for rustfmt -- LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

    RUSFMT_BIN=$1/rustfmt
    RUSTFMT_VERSION=$($RUSFMT_BIN --version)
    echo -e "\nRUSFMT_BIN $RUSTFMT_VERSION\n"

    FEATURE_BIN=$1/feature_rustfmt
    FEATURE_VERSION=$($FEATURE_BIN --version)
    echo -e "FEATURE_BIN $FEATURE_VERSION\n"
}

# Check the diff for running rustfmt and the feature branch on all the .rs files in the repo.
#
# Parameters
# $1: Clone URL for the repo
# $2: Name of the repo (mostly used for logging)
# $3: Path to any submodules that should be initialized
function check_repo() {
    WORKDIR=$(pwd)
    REPO_URL=$1
    REPO_NAME=$2
    SUBMODULES=$3

    local tmp_dir;
    tmp_dir=$(mktemp -d -t $REPO_NAME-XXXXXXXX)
    clone_repo $REPO_URL $tmp_dir

    if [ ! -z "$SUBMODULES" ]; then
        init_submodules $SUBMODULES
    fi


    # rustfmt --check returns 1 if a diff was found
    # Also check_diff returns 1 if there was a diff between master rustfmt and the feature branch
    # so we want to ignore the exit status check
    set +e
    check_diff $REPO_NAME
    # append the status of running `check_diff` to the STATUSES array
    STATUSES+=($?)
    set -e

    echo -e "removing tmp_dir $tmp_dir\n\n"
    rm -rf $tmp_dir
    cd $WORKDIR
}

function main() {
    tmp_dir=$(mktemp -d -t rustfmt-XXXXXXXX)
    echo Created tmp_dir $tmp_dir

    compile_rustfmt $tmp_dir

    # run checks
    check_repo "https://github.com/rust-lang/rust.git" rust-lang-rust
    check_repo "https://github.com/rust-lang/cargo.git" cargo
    check_repo "https://github.com/rust-lang/miri.git" miri
    check_repo "https://github.com/rust-lang/rust-analyzer.git" rust-analyzer
    check_repo "https://github.com/bitflags/bitflags.git" bitflags
    check_repo "https://github.com/rust-lang/log.git" log
    check_repo "https://github.com/rust-lang/mdBook.git" mdBook
    check_repo "https://github.com/rust-lang/packed_simd.git" packed_simd
    check_repo "https://github.com/rust-lang/rust-semverver.git" check_repo
    check_repo "https://github.com/Stebalien/tempfile.git" tempfile
    check_repo "https://github.com/rust-lang/futures-rs.git" futures-rs
    check_repo "https://github.com/dtolnay/anyhow.git" anyhow
    check_repo "https://github.com/dtolnay/thiserror.git" thiserror
    check_repo "https://github.com/dtolnay/syn.git" syn
    check_repo "https://github.com/serde-rs/serde.git" serde
    check_repo "https://github.com/rust-lang/rustlings.git" rustlings
    check_repo "https://github.com/rust-lang/rustup.git" rustup
    check_repo "https://github.com/SergioBenitez/Rocket.git" Rocket
    check_repo "https://github.com/rustls/rustls.git" rustls
    check_repo "https://github.com/rust-lang/rust-bindgen.git" rust-bindgen
    check_repo "https://github.com/hyperium/hyper.git" hyper
    check_repo "https://github.com/actix/actix.git" actix
    check_repo "https://github.com/denoland/deno.git" denoland_deno

    # cleanup temp dir
    echo removing tmp_dir $tmp_dir
    rm -rf $tmp_dir

    # figure out the exit code
    for status in ${STATUSES[@]}
    do
        if [ $status -eq 1 ]; then
            echo "formatting diff found 💔"
            return 1
        fi
    done

    echo "no diff found 😊"
}

main
