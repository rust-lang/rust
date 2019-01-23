stdsimd - Rust's standard library SIMD components
=======

[![Travis-CI Status]][travis] [![Appveyor Status]][appveyor] [![Latest Version]][crates.io] [![docs]][docs.rs]

# Crates

View the README's of:

* [`core_arch`](crates/core_arch/README.md)
* [`std_detect`](crates/std_detect/README.md)

The `std::simd` component now lives in the
[`packed_simd`](https://github.com/rust-lang-nursery/packed_simd) crate.

# How to do a release

To do a release of the `core_arch` and `std_detect` crates, 

* comment out the `dev-dependencies` in their `Cargo.toml` files (due to
  https://github.com/rust-lang/cargo/issues/4242),
* publish the crates.

[travis]: https://travis-ci.com/rust-lang-nursery/stdsimd
[Travis-CI Status]: https://travis-ci.com/rust-lang-nursery/stdsimd.svg?branch=master
[appveyor]: https://ci.appveyor.com/project/rust-lang-libs/stdsimd/branch/master
[Appveyor Status]: https://ci.appveyor.com/api/projects/status/ix74qhmilpibn00x/branch/master?svg=true
[Latest Version]: https://img.shields.io/crates/v/stdsimd.svg
[crates.io]: https://crates.io/crates/stdsimd
[docs]: https://docs.rs/stdsimd/badge.svg
[docs.rs]: https://docs.rs/stdsimd/
