`std::detect` - Rust's standard library run-time CPU feature detection
=======

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

# Features

* `std_detect_dlsym_getauxval` (enabled by default, requires `libc`): Enable to
use `libc::dlsym` to query whether [`getauxval`] is linked into the binary. When
this is not the case, this feature allows other fallback methods to perform
run-time feature detection. When this feature is disabled, `std_detect` assumes
that [`getauxval`] is linked to the binary. If that is not the case the behavior
is undefined.

* `std_detect_file_io` (enabled by default, requires `std`): Enable to perform run-time feature
detection using file APIs (e.g. `/proc/cpuinfo`, etc.) if other more performant
methods fail. This feature requires `libstd` as a dependency, preventing the
crate from working on applications in which `std` is not available.

[`getauxval`]: http://man7.org/linux/man-pages/man3/getauxval.3.html

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
