`core::arch` - Rust's core library architecture-specific intrinsics
=======

The `core::arch` module implements architecture-dependent intrinsics (e.g. SIMD).

# Usage 

`core::arch` is available as part of `libcore` and it is re-exported by
`libstd`. Prefer using it via `core::arch` or `std::arch` than via this crate.
Unstable features are often available in nightly Rust via the
`feature(stdsimd)`.

Using `core::arch` via this crate requires nightly Rust, and it can (and does)
break often. The only cases in which you should consider using it via this crate
are:

* if you need to re-compile `core::arch` yourself, e.g., with particular
  target-features enabled that are not enabled for `libcore`/`libstd`. Note: if
  you need to re-compile it for a non-standard target, please prefer using
  `xargo` and re-compiling `libcore`/`libstd` as appropriate instead of using
  this crate.
  
* using some features that might not be available even behind unstable Rust
  features. We try to keep these to a minimum. If you need to use some of these
  features, please open an issue so that we can expose them in nightly Rust and
  you can use them from there.

# Documentation

* [Documentation - i686][i686]
* [Documentation - x86\_64][x86_64]
* [Documentation - arm][arm]
* [Documentation - aarch64][aarch64]
* [Documentation - powerpc][powerpc]
* [Documentation - powerpc64][powerpc64]
* [How to get started][contrib]
* [How to help implement intrinsics][help-implement]

[contrib]: https://github.com/rust-lang/stdarch/blob/master/CONTRIBUTING.md
[help-implement]: https://github.com/rust-lang/stdarch/issues/40
[i686]: https://rust-lang.github.io/stdarch/i686/core_arch/
[x86_64]: https://rust-lang.github.io/stdarch/x86_64/core_arch/
[arm]: https://rust-lang.github.io/stdarch/arm/core_arch/
[aarch64]: https://rust-lang.github.io/stdarch/aarch64/core_arch/
[powerpc]: https://rust-lang.github.io/stdarch/powerpc/core_arch/
[powerpc64]: https://rust-lang.github.io/stdarch/powerpc64/core_arch/

# License

`core_arch` is primarily distributed under the terms of both the MIT license and
the Apache License (Version 2.0), with portions covered by various BSD-like
licenses.

See LICENSE-APACHE, and LICENSE-MIT for details.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `core_arch` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.
