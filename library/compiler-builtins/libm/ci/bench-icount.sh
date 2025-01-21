#!/bin/bash

set -eux

iai_home="iai-home"

# Download the baseline from master
./ci/ci-util.py locate-baseline --download --extract

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

    # Run iai-callgrind benchmarks
    cargo bench "${cargo_args[@]}" -- "${iai_args[@]}"

    # NB: iai-callgrind should exit on error but does not, so we inspect the sumary
    # for errors. See  https://github.com/iai-callgrind/iai-callgrind/issues/337
    if [ -n "${PR_NUMBER:-}" ]; then
        # If this is for a pull request, ignore regressions if specified.
        ./ci/ci-util.py check-regressions --home "$iai_home" --allow-pr-override "$PR_NUMBER"
    else
        ./ci/ci-util.py check-regressions --home "$iai_home" || true
    fi
}

# Run once with softfloats, once with arch instructions enabled
run_icount_benchmarks --features force-soft-floats -- --save-baseline=softfloat
run_icount_benchmarks -- --save-baseline=hardfloat

# Name and tar the new baseline
name="baseline-icount-$(date -u +'%Y%m%d%H%M')-${GITHUB_SHA:0:12}"
echo "BASELINE_NAME=$name" >>"$GITHUB_ENV"
tar cJf "$name.tar.xz" "$iai_home"
