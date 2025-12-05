//! Random value generation.

#[unstable(feature = "random", issue = "130703")]
pub use core::random::*;

use crate::sys::random as sys;

/// The default random source.
///
/// This asks the system for random data suitable for cryptographic purposes
/// such as key generation. If security is a concern, consult the platform
/// documentation below for the specific guarantees your target provides.
///
/// The high quality of randomness provided by this source means it can be quite
/// slow on some targets. If you need a large quantity of random numbers and
/// security is not a concern,  consider using an alternative random number
/// generator (potentially seeded from this one).
///
/// # Underlying sources
///
/// Platform               | Source
/// -----------------------|---------------------------------------------------------------
/// Linux                  | [`getrandom`] or [`/dev/urandom`] after polling `/dev/random`
/// Windows                | [`ProcessPrng`](https://learn.microsoft.com/en-us/windows/win32/seccng/processprng)
/// Apple                  | `CCRandomGenerateBytes`
/// DragonFly              | [`arc4random_buf`](https://man.dragonflybsd.org/?command=arc4random)
/// ESP-IDF                | [`esp_fill_random`](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/system/random.html#_CPPv415esp_fill_randomPv6size_t)
/// FreeBSD                | [`arc4random_buf`](https://man.freebsd.org/cgi/man.cgi?query=arc4random)
/// Fuchsia                | [`cprng_draw`](https://fuchsia.dev/reference/syscalls/cprng_draw)
/// Haiku                  | `arc4random_buf`
/// Illumos                | [`arc4random_buf`](https://www.illumos.org/man/3C/arc4random)
/// NetBSD                 | [`arc4random_buf`](https://man.netbsd.org/arc4random.3)
/// OpenBSD                | [`arc4random_buf`](https://man.openbsd.org/arc4random.3)
/// Solaris                | [`arc4random_buf`](https://docs.oracle.com/cd/E88353_01/html/E37843/arc4random-3c.html)
/// Vita                   | `arc4random_buf`
/// Hermit                 | `read_entropy`
/// Horizon, Cygwin        | `getrandom`
/// AIX, Hurd, L4Re, QNX   | `/dev/urandom`
/// Redox                  | `/scheme/rand`
/// RTEMS                  | [`arc4random_buf`](https://docs.rtems.org/branches/master/bsp-howto/getentropy.html)
/// SGX                    | [`rdrand`](https://en.wikipedia.org/wiki/RDRAND)
/// SOLID                  | `SOLID_RNG_SampleRandomBytes`
/// TEEOS                  | `TEE_GenerateRandom`
/// UEFI                   | [`EFI_RNG_PROTOCOL`](https://uefi.org/specs/UEFI/2.10/37_Secure_Technologies.html#random-number-generator-protocol)
/// VxWorks                | `randABytes` after waiting for `randSecure` to become ready
/// WASI                   | [`random_get`](https://github.com/WebAssembly/WASI/blob/main/legacy/preview1/docs.md#-random_getbuf-pointeru8-buf_len-size---result-errno)
/// ZKVM                   | `sys_rand`
///
/// Note that the sources used might change over time.
///
/// Consult the documentation for the underlying operations on your supported
/// targets to determine whether they provide any particular desired properties,
/// such as support for reseeding on VM fork operations.
///
/// [`getrandom`]: https://www.man7.org/linux/man-pages/man2/getrandom.2.html
/// [`/dev/urandom`]: https://www.man7.org/linux/man-pages/man4/random.4.html
#[derive(Default, Debug, Clone, Copy)]
#[unstable(feature = "random", issue = "130703")]
pub struct DefaultRandomSource;

#[unstable(feature = "random", issue = "130703")]
impl RandomSource for DefaultRandomSource {
    fn fill_bytes(&mut self, bytes: &mut [u8]) {
        sys::fill_bytes(bytes)
    }
}

/// Generates a random value from a distribution, using the default random source.
///
/// This is a convenience function for `dist.sample(&mut DefaultRandomSource)` and will sample
/// according to the same distribution as the underlying [`Distribution`] trait implementation. See
/// [`DefaultRandomSource`] for more information about how randomness is sourced.
///
/// # Examples
///
/// Generating a [version 4/variant 1 UUID] represented as text:
/// ```
/// #![feature(random)]
///
/// use std::random::random;
///
/// let bits: u128 = random(..);
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
#[unstable(feature = "random", issue = "130703")]
pub fn random<T>(dist: impl Distribution<T>) -> T {
    dist.sample(&mut DefaultRandomSource)
}
