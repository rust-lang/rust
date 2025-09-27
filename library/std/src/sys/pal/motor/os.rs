use super::map_motor_error;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::path::{self, PathBuf};
use crate::{fmt, io};

pub fn errno() -> i32 {
    // Not used in Motor OS.
    -1
}

pub fn error_string(errno: i32) -> String {
    format!("errno: {errno}")
}

pub fn getcwd() -> io::Result<PathBuf> {
    // The CWD is a bad/outdated/unix design, from a single-threaded era:
    // concurrent changes to CWD lead to races. Applications/processes
    // should manage their CWD, not the OS.
    moto_rt::fs::getcwd().map(|s| -> PathBuf { s.into() }).map_err(map_motor_error)
}

pub fn chdir(path: &path::Path) -> io::Result<()> {
    // The CWD is a bad/outdated/unix design, from a single-threaded era:
    // concurrent changes to CWD lead to races. Applications/processes
    // should manage their CWD, not the OS.
    if let Some(path) = path.to_str() {
        moto_rt::fs::chdir(path).map_err(map_motor_error)
    } else {
        Err(io::Error::new(io::ErrorKind::InvalidFilename, ""))
    }
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
    Ok(crate::sys::args::args().next().unwrap().into())
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from(moto_rt::fs::TEMP_DIR)
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    moto_rt::process::exit(code)
}

pub fn getpid() -> u32 {
    // Our pids are u64. Why does Rust mandate u32???
    panic!("no pids on this platform")
}
