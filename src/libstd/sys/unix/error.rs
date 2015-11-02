use ffi::{CStr, NulError};
use libc;
use fmt;
use mem;
use borrow::Cow;
use io::ErrorKind;
use core::nonzero::NonZero;

type ErrorCode = NonZero<i32>;

pub struct Error(ErrorCode);
pub struct ErrorString([libc::c_char; TMPBUF_SZ], usize);

pub use sys::common::error::{Result, expect_last_result, expect_last_error};

impl Error {
    pub fn from_code(code: i32) -> Self {
        debug_assert!(code != 0);
        unsafe { Error(NonZero::new(code)) }
    }

    pub fn last_error() -> Option<Self> {
        Some(Error::from_code(errno()))
    }

    pub fn default() -> Self {
        Error::from_code(libc::EINVAL)
    }

    pub fn code(&self) -> i32 { *self.0 }

    pub fn description(&self) -> ErrorString {
        // TODO: gai_strerror
        let mut s = [0 as libc::c_char; TMPBUF_SZ];
        let len = error_string(self.code(), &mut s);
        ErrorString(s, len.unwrap_or(0))
    }

    pub fn kind(&self) -> ErrorKind {
        match errno as libc::c_int {
            libc::ECONNREFUSED => ErrorKind::ConnectionRefused,
            libc::ECONNRESET => ErrorKind::ConnectionReset,
            libc::EPERM | libc::EACCES => ErrorKind::PermissionDenied,
            libc::EPIPE => ErrorKind::BrokenPipe,
            libc::ENOTCONN => ErrorKind::NotConnected,
            libc::ECONNABORTED => ErrorKind::ConnectionAborted,
            libc::EADDRNOTAVAIL => ErrorKind::AddrNotAvailable,
            libc::EADDRINUSE => ErrorKind::AddrInUse,
            libc::ENOENT => ErrorKind::NotFound,
            libc::EINTR => ErrorKind::Interrupted,
            libc::EINVAL => ErrorKind::InvalidInput,
            libc::ETIMEDOUT => ErrorKind::TimedOut,
            libc::consts::os::posix88::EEXIST => ErrorKind::AlreadyExists,

            // These two constants can have the same value on some systems,
            // but different values on others, so we can't use a match
            // clause
            x if x == libc::EAGAIN || x == libc::EWOULDBLOCK =>
                ErrorKind::WouldBlock,
            _ => ErrorKind::Other,
        }
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Error::from_code(libc::EINVAL)
    }
}

impl From<fmt::Error> for Error {
    fn from(_: fmt::Error) -> Self {
        Error::from_code(libc::EIO)
    }
}

impl ErrorString {
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { mem::transmute(&self.0[..self.1]) }
    }

    pub fn to_string_lossy(&self) -> Cow<str> {
        use string::String;

        String::from_utf8_lossy(self.as_bytes())
    }
}

const TMPBUF_SZ: usize = 128;

/// Returns the platform-specific value of errno
fn errno() -> i32 {
    extern {
        #[cfg_attr(any(target_os = "linux", target_os = "android"), link_name = "__errno_location")]
        #[cfg_attr(any(target_os = "bitrig", target_os = "netbsd", target_os = "openbsd",
                       target_env = "newlib"),
                       link_name = "__errno")]
        #[cfg_attr(target_os = "dragonfly", link_name = "__dfly_error")]
        #[cfg_attr(any(target_os = "macos", target_os = "ios", target_os = "freebsd"),
                   link_name = "__error")]
        fn errno_location() -> *const libc::c_int;
    }

    unsafe {
        (*errno_location()) as i32
    }
}

/// Gets a detailed string description for the given error number.
fn error_string(errno: i32, buf: &mut [libc::c_char]) -> ::result::Result<usize, Error> {
    extern {
        #[cfg_attr(any(target_os = "linux", target_env = "newlib"),
                   link_name = "__xpg_strerror_r")]
        fn strerror_r(errnum: libc::c_int, buf: *mut libc::c_char,
                      buflen: libc::size_t) -> libc::c_int;
    }

    let p = buf.as_mut_ptr();
    unsafe {
        match strerror_r(errno as libc::c_int, p, buf.len() as libc::size_t) {
            0 => Ok(CStr::from_ptr(p as *const _).to_bytes().len()),
            e => Err(Error::from_code(e)),
        }
    }
}
