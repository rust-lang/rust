stdarch - Rust's standard library SIMD components
=======

[![Actions Status](https://github.com/rust-lang/stdarch/workflows/CI/badge.svg)](https://github.com/rust-lang/stdarch/actions)


This repository contains the [`core_arch`](crates/core_arch/README.md) crate, which implements `core::arch` - Rust's core library architecture-specific intrinsics.
  
The `std::simd` component now lives in the
[`packed_simd_2`](https://github.com/rust-lang/packed_simd) crate.

## Synchronizing josh subtree with rustc

This repository is linked to `rust-lang/rust` as a [josh](https://josh-project.github.io/josh/intro.html) subtree. You can use the [rustc-josh-sync](https://github.com/rust-lang/josh-sync) tool to perform synchronization.

You can find a guide on how to perform the synchronization [here](https://rustc-dev-guide.rust-lang.org/external-repos.html#synchronizing-a-josh-subtree).
