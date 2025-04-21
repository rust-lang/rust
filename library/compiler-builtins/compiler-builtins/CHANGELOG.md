# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.156](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.155...compiler_builtins-v0.1.156) - 2025-04-21

### Other

- avr: Provide `abort()`
- Remove `unsafe` from `naked_asm!` blocks
- Enable icount benchmarks in CI
- Move builtins-test-intrinsics out of the workspace
- Run `cargo fmt` on all projects
- Flatten the `libm/libm` directory
- Update path to libm after the merge

## [0.1.155](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.154...compiler_builtins-v0.1.155) - 2025-04-17

### Other

- use `#[cfg(bootstrap)]` for rustc sync
- Replace the `bl!` macro with `asm_sym`
- __udivmod(h|q)i4

## [0.1.154](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.153...compiler_builtins-v0.1.154) - 2025-04-16

### Other

- turn #[naked] into an unsafe attribute

## [0.1.153](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.152...compiler_builtins-v0.1.153) - 2025-04-09

### Other

- Remove a mention of `force-soft-float` in `build.rs`
- Revert "Disable `f16` on AArch64 without the `neon` feature"
- Skip No More!
- avoid out-of-bounds accesses ([#799](https://github.com/rust-lang/compiler-builtins/pull/799))

## [0.1.152](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.151...compiler_builtins-v0.1.152) - 2025-03-20

### Other

- Remove use of `atomic_load_unordered` and undefined behaviour from `arm_linux.rs`
- Switch repository layout to use a virtual manifest
