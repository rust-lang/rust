use super::{BorrowedBuf, BorrowedCursor, Read, Result};
use crate::sys::entropy as sys;

/// A reader which returns random bytes from the system entropy source.
///
/// This struct is generally created by calling [`entropy()`]. Please
/// see the documentation of [`entropy()`] for more details.
#[derive(Debug)]
#[unstable(feature = "io_entropy", issue = "none")]
pub struct Entropy {
    insecure: bool,
}

impl Entropy {
    pub(crate) fn set_insecure(&mut self, insecure: bool) {
        self.insecure = insecure;
    }
}

#[unstable(feature = "io_entropy", issue = "none")]
impl Read for Entropy {
    #[inline]
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        sys::Entropy { insecure: self.insecure }.read(buf)
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        let mut buf = BorrowedBuf::from(buf);
        self.read_buf_exact(buf.unfilled())
    }

    #[inline]
    fn read_buf(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
        sys::Entropy { insecure: self.insecure }.read_buf(buf)
    }

    #[inline]
    fn read_buf_exact(&mut self, buf: BorrowedCursor<'_>) -> Result<()> {
        sys::Entropy { insecure: self.insecure }.read_buf_exact(buf)
    }
}

/// Constructs a new handle to the system entropy source.
///
/// Reads from the resulting reader will return high-quality random data that
/// is suited for cryptographic purposes (by the standard of the platform defaults).
///
/// Be aware that, because the data is of very high quality, reading high amounts
/// of data can be very slow, and potentially slow down other processes requiring
/// random data. Use a pseudo-random number generator if speed is important.
///
/// # Platform sources
///
/// | OS               | Source
/// |------------------|--------
/// | Linux, Android   | [`getrandom`][1] if available, otherwise [`/dev/urandom`][2] after successfully polling [`/dev/random`][2]
/// | Windows          | [`BCryptGenRandom`][3], falling back to [`RtlGenRandom`][4]
/// | macOS            | [`getentropy`][5] if available, falling back to [`/dev/urandom`][6]
/// | OpenBSD          | [`getentropy`][7]
/// | iOS, watchOS     | [`SecRandomCopyBytes`][8]
/// | FreeBSD          | [`kern.arandom`][9]
/// | NetBSD           | [`kern.arandom`][10]
/// | Fuchsia          | [`zx_cprng_draw`][11]
/// | WASM             | *Unsupported*
/// | WASI             | [`random_get`][12]
/// | Emscripten       | [`getentropy`][7]
/// | Redox            | `rand:`
/// | VxWorks          | `randABytes` after checking entropy pool initialization with `randSecure`
/// | Haiku            | `/dev/urandom`
/// | ESP-IDF, Horizon | [`getrandom`][1]
/// | Other UNIXes     | `/dev/random`
/// | Hermit           | [`read_entropy`][13]
/// | SGX              | [`RDRAND`][14]
/// | SOLID            | `SOLID_RNG_SampleRandomBytes`
///
/// [1]: https://man7.org/linux/man-pages/man2/getrandom.2.html
/// [2]: https://man7.org/linux/man-pages/man7/random.7.html
/// [3]: https://learn.microsoft.com/en-us/windows/win32/api/bcrypt/nf-bcrypt-bcryptgenrandom
/// [4]: https://learn.microsoft.com/en-us/windows/win32/api/ntsecapi/nf-ntsecapi-rtlgenrandom
/// [5]: https://www.unix.com/man-page/mojave/2/getentropy/
/// [6]: https://www.unix.com/man-page/mojave/4/random/
/// [7]: https://man.openbsd.org/getentropy.2
/// [8]: https://developer.apple.com/documentation/security/1399291-secrandomcopybytes?language=objc
/// [9]: https://man.freebsd.org/cgi/man.cgi?query=random&sektion=4&manpath=FreeBSD+13.1-RELEASE+and+Ports
/// [10]: https://man.netbsd.org/rnd.4
/// [11]: https://fuchsia.dev/fuchsia-src/reference/syscalls/cprng_draw
/// [12]: https://github.com/WebAssembly/WASI/blob/main/legacy/preview1/docs.md
/// [13]: https://docs.rs/hermit-abi/latest/hermit_abi/fn.read_entropy.html
/// [14]: https://www.intel.com/content/www/us/en/developer/articles/guide/intel-digital-random-number-generator-drng-software-implementation-guide.html
///
/// # Examples
///
/// Generating a seed for a random number generator:
///
/// ```rust
/// #![feature(io_entropy)]
///
/// # use std::io::Result;
/// # fn main() -> Result<()> {
/// use std::io::{entropy, Read};
///
/// let mut seed = [0u8; 32];
/// entropy().read_exact(&mut seed)?;
/// println!("seed: {seed:?}");
/// # Ok(())
/// # }
/// ```
///
/// Implementing your very own `/dev/random`:
///
/// ```rust, no_run
/// #![feature(io_entropy)]
///
/// use std::io::{copy, entropy, stdout};
///
/// fn main() {
///     let _ = copy(&mut entropy(), &mut stdout());
/// }
/// ```
#[inline]
#[unstable(feature = "io_entropy", issue = "none")]
pub fn entropy() -> Entropy {
    Entropy { insecure: false }
}
