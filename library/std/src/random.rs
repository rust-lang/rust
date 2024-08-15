//! Random value generation.
//!
//! The [`Random`] trait allows generating a random value for a type using a
//! given [`RandomSource`].

#[unstable(feature = "random", issue = "none")]
pub use core::random::*;

use crate::sys::random as sys;

/// The default random source.
///
/// This asks the system for random data suitable for cryptographic purposes
/// such as key generation. If security is a concern, consult the platform
/// documentation below for the specific guarantees your target provides.
///
/// The high quality of randomness provided by this source means it can be quite
/// slow. If you need a large quantity of random numbers and security is not a
/// concern,  consider using an alternative random number generator (potentially
/// seeded from this one).
///
/// # Underlying sources
///
/// Platform               | Source
/// -----------------------|---------------------------------------------------------------
/// Linux                  | [`getrandom`] or [`/dev/urandom`] after polling `/dev/random`
/// Windows                | [`ProcessPrng`]
/// macOS and other UNIXes | [`getentropy`]
/// other Apple platforms  | `CCRandomGenerateBytes`
/// ESP-IDF                | [`esp_fill_random`]
/// Fuchsia                | [`cprng_draw`]
/// Hermit                 | `read_entropy`
/// Horizon                | `getrandom` shim
/// Hurd, L4Re, QNX        | `/dev/urandom`
/// NetBSD before 10.0     | [`kern.arandom`]
/// Redox                  | `/scheme/rand`
/// SGX                    | [`rdrand`]
/// SOLID                  | `SOLID_RNG_SampleRandomBytes`
/// TEEOS                  | `TEE_GenerateRandom`
/// UEFI                   | [`EFI_RNG_PROTOCOL`]
/// VxWorks                | `randABytes` after waiting for `randSecure` to become ready
/// WASI                   | `random_get`
/// ZKVM                   | `sys_rand`
///
/// **Disclaimer:** The sources used might change over time.
///
/// [`getrandom`]: https://www.man7.org/linux/man-pages/man2/getrandom.2.html
/// [`/dev/urandom`]: https://www.man7.org/linux/man-pages/man4/random.4.html
/// [`ProcessPrng`]: https://learn.microsoft.com/en-us/windows/win32/seccng/processprng
/// [`getentropy`]: https://pubs.opengroup.org/onlinepubs/9799919799/functions/getentropy.html
/// [`esp_fill_random`]: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/random.html#_CPPv415esp_fill_randomPv6size_t
/// [`cprng_draw`]: https://fuchsia.dev/reference/syscalls/cprng_draw
/// [`kern.arandom`]: https://man.netbsd.org/rnd.4
/// [`rdrand`]: https://en.wikipedia.org/wiki/RDRAND
/// [`EFI_RNG_PROTOCOL`]: https://uefi.org/specs/UEFI/2.10/37_Secure_Technologies.html#random-number-generator-protocol
#[derive(Default, Debug, Clone, Copy)]
#[unstable(feature = "random", issue = "none")]
pub struct DefaultRandomSource;

#[unstable(feature = "random", issue = "none")]
impl RandomSource for DefaultRandomSource {
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        sys::fill_bytes(bytes)
    }
}

/// Generates a random value with the default random source.
///
/// This is a convenience function for `T::random(&mut DefaultRandomSource)` and
/// will sample according to the same distribution as the underlying [`Random`]
/// trait implementation.
///
/// **Warning:** Be careful when manipulating random values! The
/// [`random`](Random::random) method on integers samples them with a uniform
/// distribution, so a value of 1 is just as likely as [`i32::MAX`]. By using
/// modulo operations, some of the resulting values can become more likely than
/// others. Use audited crates when in doubt.
///
/// # Examples
///
/// Generating a [version 4/variant 1 UUID] represented as text:
/// ```
/// #![feature(random)]
///
/// use std::random::random;
///
/// let bits = random::<u128>();
/// let g1 = (bits >> 96) as u32;
/// let g2 = (bits >> 80) as u16;
/// let g3 = (0x4000 | (bits >> 64) & 0x0fff) as u16;
/// let g4 = (0x8000 | (bits >> 48) & 0x3fff) as u16;
/// let g5 = (bits & 0xffffffffffff) as u64;
/// let uuid = format!("{g1:08x}-{g2:04x}-{g3:04x}-{g4:04x}-{g5:012x}");
/// println!("{uuid}");
/// ```
///
/// [version 4/variant 1 UUID]: https://en.wikipedia.org/wiki/Universally_unique_identifier#Version_4_(random)
pub fn random<T: Random>() -> T {
    T::random(&mut DefaultRandomSource)
}
