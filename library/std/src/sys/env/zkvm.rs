use crate::ffi::{OsStr, OsString};
use crate::sys::os_str;
use crate::sys::pal::{WORD_SIZE, abi};
use crate::sys_common::FromInner;
use crate::{fmt, io};

pub struct Env(!);

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.0
    }
}

pub fn env() -> Env {
    panic!("not supported on this platform")
}

impl Env {
    pub fn str_debug(&self) -> impl fmt::Debug + '_ {
        let Self(inner) = self;
        match *inner {}
    }
}

impl fmt::Debug for Env {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Self(inner) = self;
        match *inner {}
    }
}

pub fn getenv(varname: &OsStr) -> Option<OsString> {
    let varname = varname.as_encoded_bytes();
    let nbytes =
        unsafe { abi::sys_getenv(crate::ptr::null_mut(), 0, varname.as_ptr(), varname.len()) };
    if nbytes == usize::MAX {
        return None;
    }

    let nwords = (nbytes + WORD_SIZE - 1) / WORD_SIZE;
    let words = unsafe { abi::sys_alloc_words(nwords) };

    let nbytes2 = unsafe { abi::sys_getenv(words, nwords, varname.as_ptr(), varname.len()) };
    debug_assert_eq!(nbytes, nbytes2);

    // Convert to OsString.
    //
    // FIXME: We can probably get rid of the extra copy here if we
    // reimplement "os_str" instead of just using the generic unix
    // "os_str".
    let u8s: &[u8] = unsafe { crate::slice::from_raw_parts(words.cast() as *const u8, nbytes) };
    Some(OsString::from_inner(os_str::Buf { inner: u8s.to_vec() }))
}

pub unsafe fn setenv(_: &OsStr, _: &OsStr) -> io::Result<()> {
    Err(io::const_error!(io::ErrorKind::Unsupported, "cannot set env vars on this platform"))
}

pub unsafe fn unsetenv(_: &OsStr) -> io::Result<()> {
    Err(io::const_error!(io::ErrorKind::Unsupported, "cannot unset env vars on this platform"))
}
