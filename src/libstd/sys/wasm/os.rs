use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::path::{self, PathBuf};
use crate::str;
use crate::sys::{unsupported, Void};

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

pub struct SplitPaths<'a>(&'a Void);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        match *self.0 {}
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
        "not supported on wasm yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported on wasm yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub struct Env(Void);

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        match self.0 {}
    }
}

pub fn env() -> Env {
    panic!("not supported on web assembly")
}

pub fn getenv(_: &OsStr) -> io::Result<Option<OsString>> {
    Ok(None)
}

pub fn setenv(_: &OsStr, _: &OsStr) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Other, "cannot set env vars on wasm32-unknown-unknown"))
}

pub fn unsetenv(_: &OsStr) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Other, "cannot unset env vars on wasm32-unknown-unknown"))
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on wasm")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(_code: i32) -> ! {
    unsafe {
        crate::arch::wasm32::unreachable();
    }
}

pub fn getpid() -> u32 {
    panic!("no pids on wasm")
}
