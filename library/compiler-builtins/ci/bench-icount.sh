#!/bin/bash

set -eux

target="${1:-}"

if [ -z "$target" ]; then
    host_target=$(rustc -vV | awk '/^host/ { print $2 }')
    echo "Defaulted to host target $host_target"
    target="$host_target"
fi

iai_home="iai-home"

# Use the arch as a tag to disambiguate artifacts
tag="$(echo "$target" | cut -d'-' -f1)"

# Download the baseline from master
./ci/ci-util.py locate-baseline --download --extract --tag "$tag"

# Run benchmarks once
function run_icount_benchmarks() {
    cargo_args=(
        "--bench" "icount"
        "--no-default-features"
        "--features" "unstable,unstable-float,icount"
    )

    iai_args=(
        "--home" "$(pwd)/$iai_home"
        "--regression=ir=5.0"
        "--save-summary"
    )

    # Parse `cargo_arg0 cargo_arg1 -- iai_arg0 iai_arg1` syntax
    parsing_iai_args=0
    while [ "$#" -gt 0 ]; do
        if [ "$parsing_iai_args" == "1" ]; then
            iai_args+=("$1")
        elif [ "$1" == "--" ]; then
            parsing_iai_args=1
        else
            cargo_args+=("$1")
        fi

        shift
    done

    # Run iai-callgrind benchmarks. Do this in a subshell with `&& true` to
    # capture rather than exit on error.
    (cargo bench "${cargo_args[@]}" -- "${iai_args[@]}") && true
    exit_code="$?"

    if [ "$exit_code" -eq 0 ]; then
        echo "Benchmarks completed with no regressions"
    elif [ -z "${PR_NUMBER:-}" ]; then
        # Disregard regressions after merge
        echo "Benchmarks completed with regressions; ignoring (not in a PR)"
    else
        ./ci/ci-util.py handle-banch-regressions "$PR_NUMBER"
    fi
}

# Run once with softfloats, once with arch instructions enabled
run_icount_benchmarks --features force-soft-floats -- --save-baseline=softfloat
run_icount_benchmarks -- --save-baseline=hardfloat

# Name and tar the new baseline
name="baseline-icount-$tag-$(date -u +'%Y%m%d%H%M')-${GITHUB_SHA:0:12}"
echo "BASELINE_NAME=$name" >>"$GITHUB_ENV"
tar cJf "$name.tar.xz" "$iai_home"
