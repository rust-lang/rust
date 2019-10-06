stdarch - Rust's standard library SIMD components
=======

# Crates

This repository contains two main crates:

* [![core_arch_crate_badge]][core_arch_crate_link]
  [![core_arch_docs_badge]][core_arch_docs_link]
  [`core_arch`](crates/core_arch/README.md) implements `core::arch` - Rust's
  core library architecture-specific intrinsics, and
  
* [![std_detect_crate_badge]][std_detect_crate_link]
  [![std_detect_docs_badge]][std_detect_docs_link]
  [`std_detect`](crates/std_detect/README.md) implements `std::detect` - Rust's
  standard library run-time CPU feature detection.

The `std::simd` component now lives in the
[`packed_simd`](https://github.com/rust-lang-nursery/packed_simd) crate.

# How to do a release

To do a release of the `core_arch` and `std_detect` crates, 

* bump up the version appropriately,
* comment out the `dev-dependencies` in their `Cargo.toml` files (due to
  https://github.com/rust-lang/cargo/issues/4242),
* publish the crates.

[core_arch_crate_badge]: https://img.shields.io/crates/v/core_arch.svg
[core_arch_crate_link]: https://crates.io/crates/core_arch
[core_arch_docs_badge]: https://docs.rs/core_arch/badge.svg
[core_arch_docs_link]: https://docs.rs/core_arch/
[std_detect_crate_badge]: https://img.shields.io/crates/v/std_detect.svg
[std_detect_crate_link]: https://crates.io/crates/std_detect
[std_detect_docs_badge]: https://docs.rs/std_detect/badge.svg
[std_detect_docs_link]: https://docs.rs/std_detect/
