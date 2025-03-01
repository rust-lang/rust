# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
