use error as sys;
use c::prelude as c;
use error::{Error as sys_Error, ErrorString as sys_ErrorString};
use c_str::{CStr, NulError};
use libc;
use core::fmt;
use core::mem;
use core::nonzero::NonZero;

pub type ErrorCode = NonZero<i32>;

pub struct Error(ErrorCode);
pub struct ErrorString([libc::c_char; TMPBUF_SZ], usize);

impl sys::Error for Error {
    type ErrorString = ErrorString;

    fn from_code(code: i32) -> Self {
        debug_assert!(code != 0);
        unsafe { Error(NonZero::new(code)) }
    }

    fn last_error() -> Option<Self> {
        Some(Error::from_code(errno()))
    }

    fn code(&self) -> i32 { *self.0 }
    fn description(&self) -> ErrorString {
        // TODO: gai_strerror
        let mut s = [0 as libc::c_char; TMPBUF_SZ];
        let len = error_string(self.code(), &mut s);
        ErrorString(s, len.unwrap_or(0))
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Debug::fmt(&(self.code(), self.description().to_str().unwrap_or("unknown error")), f)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self.description().to_str().unwrap_or("unknown error"), f)
    }
}

impl From<NulError> for Error {
    fn from(_: NulError) -> Self {
        Error::from_code(c::EINVAL)
    }
}

impl From<fmt::Error> for Error {
    fn from(_: fmt::Error) -> Self {
        Error::from_code(c::EIO)
    }
}

impl sys::ErrorString for ErrorString {
    fn as_bytes(&self) -> &[u8] {
        unsafe { mem::transmute(&self.0[..self.1]) }
    }
}

const TMPBUF_SZ: usize = 128;

/// Returns the platform-specific value of errno
fn errno() -> i32 {
    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd"))]
    unsafe fn errno_location() -> *const libc::c_int {
        extern { fn __error() -> *const libc::c_int; }
        __error()
    }

    #[cfg(target_os = "dragonfly")]
    unsafe fn errno_location() -> *const libc::c_int {
        extern { fn __dfly_error() -> *const libc::c_int; }
        __dfly_error()
    }

    #[cfg(any(target_os = "bitrig", target_os = "netbsd", target_os = "openbsd"))]
    unsafe fn errno_location() -> *const libc::c_int {
        extern { fn __errno() -> *const libc::c_int; }
        __errno()
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    unsafe fn errno_location() -> *const libc::c_int {
        extern { fn __errno_location() -> *const libc::c_int; }
        __errno_location()
    }

    unsafe {
        (*errno_location()) as i32
    }
}

/// Gets a detailed string description for the given error number.
fn error_string(errno: i32, buf: &mut [libc::c_char]) -> Result<usize, Error> {
    #[cfg(target_os = "linux")]
    extern {
        #[link_name = "__xpg_strerror_r"]
        fn strerror_r(errnum: libc::c_int, buf: *mut libc::c_char,
                      buflen: libc::size_t) -> libc::c_int;
    }
    #[cfg(not(target_os = "linux"))]
    extern {
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
