# ThingOS Build System
# Root task runner for the ThingOS forked toolchain and OS workspace.
#
# This repository is the Rust tree now, so the old `vendor/rust` fetch/reset
# flow is intentionally omitted. ThingOS-specific automation is expected to
# live in a root `xtask` crate as the project evolves.

# Target architecture to build for. Default to x86_64.
karch := env_var_or_default("KARCH", "x86_64")

# Default user QEMU flags.
qemuflags := env_var_or_default("QEMUFLAGS", "-m 2G -smp 6")

# Rust profile (dev/release).
rust_profile := env_var_or_default("RUST_PROFILE", "dev")

# Explicit xtask entry point. Avoid relying on a local `cargo xtask` alias.
xtask := "cargo run -p xtask --"

# Default target.
default: iso

# Check UI split.
check-ui-split:
    ./scripts/ci_check_ui_split.sh

# Audit platform boundary (verify no_std compliance).
audit-platform:
    {{xtask}} audit

# Alias for iso.
build arch=karch:
    @just iso {{arch}}

# Build everything (ISO) - optionally specify architecture.
# Examples: `just iso`, `just iso aarch64`
iso arch=karch:
    {{xtask}} iso --env {{arch}} --profile {{rust_profile}}

# Build HDD image.
hdd arch=karch:
    {{xtask}} hdd --env {{arch}} --profile {{rust_profile}}

# Run with QEMU (UEFI mode).
# Examples:
#   `just run`
#   `just run aarch64`
#   `just run -i`
#   `just run x86_64 -i`
run *args:
    #!/usr/bin/env bash
    set -euo pipefail
    ARCH="{{karch}}"
    ARGS_ARRAY=({{args}})
    if [[ "${ARGS_ARRAY[0]-}" != "" && "${ARGS_ARRAY[0]}" != -* ]]; then
        ARCH="${ARGS_ARRAY[0]}"
        ARGS_ARRAY=("${ARGS_ARRAY[@]:1}")
    fi
    if [[ " ${ARGS_ARRAY[*]} " != *" -i "* && " ${ARGS_ARRAY[*]} " != *" --interactive "* ]]; then
        ARGS_ARRAY+=("-i")
    fi
    RUSTFLAGS="-Awarnings" {{xtask}} run --env "$ARCH" --profile "{{rust_profile}}" "${ARGS_ARRAY[@]}" --qemu-flags "{{qemuflags}}"

# Start HTTPS proxy for guest (runs on port 8081).
# Guest accesses via: `http://10.0.2.2:8081/?url=https://example.com/`
proxy port="8081":
    {{xtask}} guest-proxy --port {{port}}

# Start HTTPS reverse proxy for host browser access to guest anther.
# Wraps guest's HTTP server (port 8888) with self-signed HTTPS.
# Browser accesses: `https://localhost:8443/`
https-proxy port="8443" target="8888":
    {{xtask}} https-proxy --port {{port}} --target {{target}}

# Run QEMU with proxy (starts proxy in background, then QEMU).
run-with-proxy arch=karch port="8081":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Starting HTTPS proxy on port {{port}}..."
    {{xtask}} guest-proxy --port {{port}} &
    PROXY_PID=$!
    trap "kill $PROXY_PID 2>/dev/null" EXIT
    echo "Proxy PID: $PROXY_PID"
    RUSTFLAGS="-Awarnings" {{xtask}} run --env {{arch}} --profile "{{rust_profile}}" --qemu-flags "{{qemuflags}}"

# Run HDD with QEMU.
run-hdd *args:
    #!/usr/bin/env bash
    set -euo pipefail
    ARCH="{{karch}}"
    if [[ "${1-}" != "" && "${1}" != -* ]]; then
        ARCH="$1"
        shift
    fi
    {{xtask}} run-hdd --env "$ARCH" --profile "{{rust_profile}}" "$@" --qemu-flags "{{qemuflags}}"

# Run with BIOS (x86_64 only).
run-bios:
    {{xtask}} run-bios --qemu-flags "{{qemuflags}}"

# Build the kernel.
kernel arch=karch:
    {{xtask}} build --env {{arch}} --profile {{rust_profile}}

# Clone and build limine bootloader.
limine:
    {{xtask}} limine

# Download OVMF firmware (all architectures by default).
ovmf:
    {{xtask}} ovmf-all

# Clean build artifacts plus fetched/vendor state.
clean:
    {{xtask}} clean

# Compatibility alias for clean.
distclean:
    {{xtask}} distclean

# Run BDD tests (default: all architectures).
# Examples:
#   `just behave`
#   `just behave --arch x86_64`
#   `just behave --feature simple-boot`
#   `just behave --tags @smoke`
behave *args:
    RUSTFLAGS="-Awarnings" {{xtask}} bdd {{args}}

# Alias for behave.
bdd *args:
    @just behave {{args}}

# Clear all BDD behavior reports (preserves .feature files and top-level README).
clear-behavior:
    rm -rf docs/behavior/x86_64 docs/behavior/aarch64 docs/behavior/riscv64 docs/behavior/loongarch64

# Kill all running QEMU instances.
die:
    {{xtask}} kill

# Build sprout user app.
sprout arch=karch:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET_ARCH="{{arch}}"
    if [[ "$TARGET_ARCH" == "riscv64" ]]; then
        TARGET_JSON="targets/riscv64gc-unknown-thingos.json"
    else
        TARGET_JSON="targets/${TARGET_ARCH}-unknown-thingos.json"
    fi
    export __CARGO_TESTS_ONLY_SRC_ROOT="$(pwd)/library"
    echo "Building sprout for $TARGET_ARCH using $TARGET_JSON..."
    RUSTFLAGS="-Awarnings" cargo -Z build-std=core,alloc,std,panic_abort -Z build-std-features=compiler-builtins-mem -Z json-target-spec build --target "$TARGET_JSON" -p sprout

# Build rtc_cmos user app.
rtc_cmos arch=karch:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET_ARCH="{{arch}}"
    if [[ "$TARGET_ARCH" == "riscv64" ]]; then
        TARGET_JSON="targets/riscv64gc-unknown-thingos.json"
    else
        TARGET_JSON="targets/${TARGET_ARCH}-unknown-thingos.json"
    fi
    export __CARGO_TESTS_ONLY_SRC_ROOT="$(pwd)/library"
    echo "Building rtc_cmos for $TARGET_ARCH using $TARGET_JSON..."
    RUSTFLAGS="-Awarnings" cargo -Z build-std=core,alloc,std,panic_abort -Z build-std-features=compiler-builtins-mem -Z json-target-spec build --target "$TARGET_JSON" -p rtc_cmos

# Build clock user app.
clock arch=karch:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET_ARCH="{{arch}}"
    if [[ "$TARGET_ARCH" == "riscv64" ]]; then
        TARGET_JSON="targets/riscv64gc-unknown-thingos.json"
    else
        TARGET_JSON="targets/${TARGET_ARCH}-unknown-thingos.json"
    fi
    export __CARGO_TESTS_ONLY_SRC_ROOT="$(pwd)/library"
    echo "Building clock for $TARGET_ARCH using $TARGET_JSON..."
    RUSTFLAGS="-Awarnings" cargo -Z build-std=core,alloc,std,panic_abort -Z build-std-features=compiler-builtins-mem -Z json-target-spec build --target "$TARGET_JSON" -p clock

# Build bristle user app.
bristle arch=karch:
    #!/usr/bin/env bash
    set -euo pipefail
    TARGET_ARCH="{{arch}}"
    if [[ "$TARGET_ARCH" == "riscv64" ]]; then
        TARGET_JSON="targets/riscv64gc-unknown-thingos.json"
    else
        TARGET_JSON="targets/${TARGET_ARCH}-unknown-thingos.json"
    fi
    export __CARGO_TESTS_ONLY_SRC_ROOT="$(pwd)/library"
    echo "Building bristle for $TARGET_ARCH using $TARGET_JSON..."
    RUSTFLAGS="-Awarnings" cargo -Z build-std=core,alloc,std,panic_abort -Z build-std-features=compiler-builtins-mem -Z json-target-spec build --target "$TARGET_JSON" -p bristle

# Build and cache the current stage-1 Rust bootstrap output.
# Today this is expected to produce a Linux-hosted cross-compiler plus a cached
# rustlib tree under `target/rustc-thingos/`. Set `SKIP_RUSTC_THINGOS=1` to opt out.
rustc-thingos:
    {{xtask}} rustc-thingos

# Fetch vendor assets (Limine, OVMF, fonts, icons, cursors).
fetch:
    {{xtask}} fetch

# Run all unit tests (host-testable crates only).
test *args:
    cargo test \
        -p abi \
        -p abi-macros \
        -p pciids \
        -p xtask \
        -p stem \
        -p stem-macros \
        -p llm \
        -p llm_stub \
        -p fb_common \
        -p kindc \
        {{args}}

# Check everything (compilation + UI split).
check: check-ui-split
    #!/usr/bin/env bash
    set -euo pipefail
    export __CARGO_TESTS_ONLY_SRC_ROOT="$(pwd)/library"
    cargo -Z build-std=core,alloc,std,panic_abort -Z build-std-features=compiler-builtins-mem -Z json-target-spec check --target targets/x86_64-unknown-thingos.json -p sprout

# Run smoke tests (quick boot validation).
smoke:
    {{xtask}} bdd --arch x86_64 --tags @smoke

# Generate Rust from .kind schema files.
kindc *args:
    cargo run -p kindc -- {{args}}

# Regenerate checked-in fixture output.
kindc-gen:
    cargo run -p kindc -- tools/kindc/kinds -o tools/kindc/fixtures/generated

# Check that generated Kind output is up-to-date (fails if drift is detected).
# Run `just kindc-gen` to fix.
kindc-check:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo run -p kindc -- tools/kindc/kinds -o tools/kindc/fixtures/generated
    if ! git diff --exit-code tools/kindc/fixtures/generated/; then
        echo ""
        echo "ERROR: Generated Kind output is out of date."
        echo "Run 'just kindc-gen' to regenerate, then commit the result."
        exit 1
    fi
    echo "kindc-check: generated output is up to date."
