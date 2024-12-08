use super::{WORD_SIZE, abi, unsupported};
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::path::{self, PathBuf};
use crate::sys::os_str;
use crate::sys_common::FromInner;
use crate::{fmt, io};

pub fn errno() -> i32 {
    0
}

pub fn error_string(_errno: i32) -> String {
    "operation successful".to_string()
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    unsupported()
}

pub struct SplitPaths<'a>(!, PhantomData<&'a ()>);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.0
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(_paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    Err(JoinPathsError)
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "not supported on this platform yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported on this platform yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

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

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(_code: i32) -> ! {
    crate::intrinsics::abort()
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}
