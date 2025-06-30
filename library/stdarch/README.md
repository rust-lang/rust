stdarch - Rust's standard library SIMD components
=======

[![Actions Status](https://github.com/rust-lang/stdarch/workflows/CI/badge.svg)](https://github.com/rust-lang/stdarch/actions)


# Crates

This repository contains two main crates:

* [`core_arch`](crates/core_arch/README.md) implements `core::arch` - Rust's
  core library architecture-specific intrinsics, and
  
* [`std_detect`](crates/std_detect/README.md) implements `std::detect` - Rust's
  standard library run-time CPU feature detection.

The `std::simd` component now lives in the
[`packed_simd_2`](https://github.com/rust-lang/packed_simd) crate.
