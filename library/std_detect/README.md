`std::detect` - Rust's standard library run-time CPU feature detection
=======

The private `std::detect` module implements run-time feature detection in Rust's
standard library. This allows detecting whether the CPU the binary runs on
supports certain features, like SIMD instructions.

# Usage

`std::detect` APIs are available as part of `libstd`. Prefer using it via the
standard library than through this crate. Unstable features of `std::detect` are
available on nightly Rust behind various feature-gates.

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
  application.

* Linux/Android:
  * `arm{32, 64}`, `mips{32,64}{,el}`, `powerpc{32,64}{,le}`, `loongarch{32,64}`, `s390x`:
    `std_detect` supports these on Linux by querying ELF auxiliary vectors (using `getauxval`
    when available), and if that fails, by querying `/proc/self/auxv`.
  * `arm64`: partial support for doing run-time feature detection by directly
    querying `mrs` is implemented for Linux >= 4.11, but not enabled by default.
  * `riscv{32,64}`:
    `std_detect` supports these on Linux by querying `riscv_hwprobe`, and
    by querying ELF auxiliary vectors (using `getauxval` when available).

* FreeBSD:
  * `arm32`, `powerpc64`: `std_detect` supports these on FreeBSD by querying ELF
    auxiliary vectors using `elf_aux_info`.
  * `arm64`: run-time feature detection is implemented by directly querying `mrs`.

* OpenBSD:
   * `powerpc64`: `std_detect` supports these on OpenBSD by querying ELF auxiliary
     vectors using `elf_aux_info`.
  * `arm64`: run-time feature detection is implemented by querying `sysctl`.

* Windows:
  * `arm64`: run-time feature detection is implemented by querying `IsProcessorFeaturePresent`.

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
