# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
