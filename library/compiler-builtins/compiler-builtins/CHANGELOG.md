# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.160](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.159...compiler_builtins-v0.1.160) - 2025-05-29

### Other

- Change `compiler-builtins` to edition 2024
- Remove unneeded C symbols
- Reuse `libm`'s `Caat` and `CastFrom` in `compiler-builtins`
- Reuse `MinInt` and `Int` from `libm` in `compiler-builtins`
- Update `CmpResult` to use a pointer-sized return type
- Enable `__powitf2` on MSVC
- Fix `i256::MAX`
- Add a note saying why we use `frintx` rather than `frintn`
- Typo in README.md
- Clean up unused files

## [0.1.159](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.158...compiler_builtins-v0.1.159) - 2025-05-12

### Other

- Remove cfg(bootstrap)

## [0.1.158](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.157...compiler_builtins-v0.1.158) - 2025-05-06

### Other

- Require `target_has_atomic = "ptr"` for runtime feature detection

## [0.1.157](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.156...compiler_builtins-v0.1.157) - 2025-05-03

### Other

- Use runtime feature detection for fma routines on x86

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

## [0.1.151](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.150...compiler_builtins-v0.1.151) - 2025-03-05

### Other

- Add cygwin support
- Enable `f16` for LoongArch ([#770](https://github.com/rust-lang/compiler-builtins/pull/770))
- Add __extendhfdf2 and add __truncdfhf2 test
- Remove outdated information from the readme

## [0.1.150](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.149...compiler_builtins-v0.1.150) - 2025-03-01

### Other

- Disable `f16` on AArch64 without the `neon` feature
- Update LLVM downloads to 20.1-2025-02-13

## [0.1.149](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.148...compiler_builtins-v0.1.149) - 2025-02-25

### Other

- Make a subset of `libm` symbols weakly available on all platforms

## [0.1.148](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.147...compiler_builtins-v0.1.148) - 2025-02-24

### Other

- Update the `libm` submodule
- Enable `f16` for MIPS
- Eliminate the use of `public_test_dep!` for a third time

## [0.1.147](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.146...compiler_builtins-v0.1.147) - 2025-02-19

### Other

- remove win64_128bit_abi_hack

## [0.1.146](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.145...compiler_builtins-v0.1.146) - 2025-02-06

### Other

- Expose erf{,c}{,f} from libm

## [0.1.145](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.144...compiler_builtins-v0.1.145) - 2025-02-04

### Other

- Revert "Eliminate the use of `public_test_dep!`"
- Indentation fix to please clippy
- Don't build out of line atomics support code for uefi
- Add a version to some FIXMEs that will be resolved in LLVM 20
- Remove use of the `start` feature

## [0.1.144](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.143...compiler_builtins-v0.1.144) - 2025-01-15

### Other

- Eliminate the use of `public_test_dep!`

## [0.1.143](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.142...compiler_builtins-v0.1.143) - 2025-01-15

### Other

- Use a C-safe return type for `__rust_[ui]128_*` overflowing intrinsics

## [0.1.142](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.141...compiler_builtins-v0.1.142) - 2025-01-07

### Other

- Account for optimization levels other than numbers

## [0.1.141](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.140...compiler_builtins-v0.1.141) - 2025-01-07

### Other

- Update the `libm` submodule
- Fix new `clippy::precedence` errors
- Rename `EXP_MAX` to `EXP_SAT`
- Shorten prefixes for float constants

## [0.1.140](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.139...compiler_builtins-v0.1.140) - 2024-12-26

### Other

- Disable f128 for amdgpu ([#737](https://github.com/rust-lang/compiler-builtins/pull/737))
- Fix a bug in `abs_diff`
- Disable `f16` on platforms that have recursion problems

## [0.1.139](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.138...compiler_builtins-v0.1.139) - 2024-11-03

### Other

- Remove incorrect `sparcv9` match pattern from `configure_f16_f128`

## [0.1.138](https://github.com/rust-lang/compiler-builtins/compare/compiler_builtins-v0.1.137...compiler_builtins-v0.1.138) - 2024-11-01

### Other

- Use `f16_enabled`/`f128_enabled` in `examples/intrinsics.rs` ([#724](https://github.com/rust-lang/compiler-builtins/pull/724))
- Disable `f16` for LoongArch64 ([#722](https://github.com/rust-lang/compiler-builtins/pull/722))
