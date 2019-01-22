`std_detect` - Rust's standard library `std::detect` module
=======

[![Travis-CI Status]][travis] [![Appveyor Status]][appveyor] [![Latest Version]][crates.io] [![docs]][docs.rs]


The private `std::detect` module implements run-time feature detection in Rust's
standard library. This allows detecting whether the CPU the binary runs on
supports certain features, like SIMD instructions.

# Usage 

`std::detect` APIs are available as part of `libstd`. Prefer using it via the
standard library than through this crate. Unstable features of `std::detect` are
available on nightly Rust behind the `feature(stdsimd)` feature-gate.

If you need run-time feature detection in `#[no_std]` environments, Rust `core`
library cannot help you. By design, Rust `core` is platform independent, but
performing run-time feature detection requires a certain level of cooperation
from the platform.

You can then manually include `std_detect` as a dependency to get similar
run-time feature detection support than the one offered by Rust's standard
library. We intend to make `std_detect` more flexible and configurable in this
regard to better serve the needs of `#[no_std]` targets. 

# Platform support

* All `x86`/`x86_64` targets are supported on all platforms by querying the
  `cpuid` instruction directly for the features supported by the hardware and
  the operating system. `std_detect` assumes that the binary is an user-space
  application. If you need raw support for querying `cpuid`, consider using the
  [`cupid`](https://crates.io/crates/cupid) crate.
  
* Linux:
  * `arm{32, 64}`, `mips{32,64}{,el}`, `powerpc{32,64}{,le}`: `std_detect`
    supports these on Linux by querying ELF auxiliary vectors (using `getauxval`
    when available), and if that fails, by querying `/proc/cpuinfo`. 
  * `arm64`: partial support for doing run-time feature detection by directly
    querying `mrs` is implemented for Linux >= 4.11, but not enabled by default.

* FreeBSD:
  * `arm64`: run-time feature detection is implemented by directly querying `mrs`.

# License

This project is licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or
   http://opensource.org/licenses/MIT)

at your option.

# Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in `std_detect` by you, as defined in the Apache-2.0 license,
shall be dual licensed as above, without any additional terms or conditions.

[travis]: https://travis-ci.org/rust-lang-nursery/stdsimd
[Travis-CI Status]: https://travis-ci.org/rust-lang-nursery/stdsimd.svg?branch=master
[appveyor]: https://ci.appveyor.com/project/rust-lang-libs/stdsimd/branch/master
[Appveyor Status]: https://ci.appveyor.com/api/projects/status/ix74qhmilpibn00x/branch/master?svg=true
[Latest Version]: https://img.shields.io/crates/v/std_detect.svg
[crates.io]: https://crates.io/crates/std_detect
[docs]: https://docs.rs/std_detect/badge.svg
[docs.rs]: https://docs.rs/std_detect/
