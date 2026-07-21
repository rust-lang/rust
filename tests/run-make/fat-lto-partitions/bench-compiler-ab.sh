#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

if (( $# != 3 )); then
    echo "usage: $0 OLD_RUSTC NEW_RUSTC PROJECT_DIR" >&2
    exit 2
fi

old_rustc=$(realpath "$1")
new_rustc=$(realpath "$2")
project=$(realpath "$3")
package=${PACKAGE:-mud1-rs}
binary=${BINARY:-mud1-studio}
runs=${RUNS:-3}
if [[ ! $runs =~ ^[1-9][0-9]*$ ]]; then
    echo "RUNS must be a positive integer" >&2
    exit 2
fi
old_target=${OLD_TARGET_DIR:-$project/target/fat-lto-perf-old}
new_target=${NEW_TARGET_DIR:-$project/target/fat-lto-perf-new}
results=${RESULTS:-$project/target/fat-lto-perf.tsv}
artifacts=${ARTIFACTS_DIR:-$project/target/fat-lto-perf-artifacts}
read -r -a rustc_args <<< "${RUSTC_ARGS:--Zfat-lto-partitions=16}"
trial_artifacts=()

mkdir -p "$artifacts"
printf 'compiler\trun\telapsed_s\tuser_s\tsystem_s\tmax_rss_kib\n' > "$results"

build() {
    local rustc=$1
    local target=$2
    env RUSTC="$rustc" CARGO_INCREMENTAL=0 \
        cargo rustc --locked --release --bin "$binary" --target-dir "$target" -- \
        "${rustc_args[@]}"
}

trial() {
    local compiler=$1
    local run=$2
    local rustc=$3
    local target=$4
    local artifact=$artifacts/$compiler-$run

    echo "start: compiler=$compiler run=$run"
    cargo clean --release -p "$package" --target-dir "$target"
    env RUSTC="$rustc" CARGO_INCREMENTAL=0 \
        /usr/bin/time -a -o "$results" \
        -f "$compiler\t$run\t%e\t%U\t%S\t%M" \
        cargo rustc --locked --release --bin "$binary" --target-dir "$target" -- \
        "${rustc_args[@]}"
    cp "$target/release/$binary" "$artifact"
    trial_artifacts+=("$artifact")
    sha256sum "$artifact"
}

cd "$project"
build "$old_rustc" "$old_target"
build "$new_rustc" "$new_target"

old_run=0
new_run=0
for (( pair = 1; pair <= runs; pair++ )); do
    if (( pair % 2 )); then
        order=(new old)
    else
        order=(old new)
    fi
    for compiler in "${order[@]}"; do
        if [[ $compiler == old ]]; then
            ((++old_run))
            trial old "$old_run" "$old_rustc" "$old_target"
        else
            ((++new_run))
            trial new "$new_run" "$new_rustc" "$new_target"
        fi
    done
done

reference=${trial_artifacts[0]}
if [[ ${ALLOW_NONIDENTICAL:-0} != 1 ]]; then
    for artifact in "${trial_artifacts[@]}"; do
        cmp "$reference" "$artifact"
    done
fi

awk '
    NR > 1 {
        count[$1]++
        elapsed[$1] += $3
        elapsed_sq[$1] += $3 * $3
        user[$1] += $4
        rss[$1] += $6
    }
    END {
        for (compiler in count) {
            mean = elapsed[compiler] / count[compiler]
            variance = count[compiler] > 1 \
                ? (elapsed_sq[compiler] - count[compiler] * mean * mean) / (count[compiler] - 1) \
                : 0
            printf "%s: n=%d elapsed_mean=%.3fs elapsed_sd=%.3fs " \
                "user_mean=%.3fs max_rss_mean=%.0fKiB\n", \
                compiler, count[compiler], mean, sqrt(variance), \
                user[compiler] / count[compiler], rss[compiler] / count[compiler]
            means[compiler] = mean
        }
        printf "new-vs-old elapsed: %+.3fs (%+.3f%%; negative means new is faster)\n", \
            means["new"] - means["old"], 100 * (means["new"] / means["old"] - 1)
    }
' "$results"
