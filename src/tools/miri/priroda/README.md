# Priroda

Priroda is a step-through debugger for Rust programs running under
Miri.

Current focus:

- simple CLI prototype
- single-threaded stepping with Miri's interpreter
- commands: empty Enter, `s`, or `step`

## Setup

From `miri/`, install the pinned toolchain and the local `cargo-miri`
command:

```sh
./miri toolchain
./miri install
```

Then build the Miri sysroot and export it for Priroda:

```sh
cargo +miri miri setup
export MIRI_SYSROOT="$(cargo +miri miri setup --print-sysroot)"
```

## Run

Priroda currently reads `MIRI_SYSROOT` directly. After setup:

```sh
cargo run -p priroda -- tests/pass/empty_main.rs
```

At the prompt, press Enter or type `s` / `step`.
