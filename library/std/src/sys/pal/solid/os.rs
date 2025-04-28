use super::{error, itron, unsupported};
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::path::{self, PathBuf};
use crate::{fmt, io};

// `solid` directly maps `errno`s to μITRON error codes.
impl itron::error::ItronError {
    #[inline]
    pub(crate) fn as_io_error(self) -> crate::io::Error {
        crate::io::Error::from_raw_os_error(self.as_raw())
    }
}

pub fn errno() -> i32 {
    0
}

pub fn error_string(errno: i32) -> String {
    if let Some(name) = error::error_name(errno) { name.to_owned() } else { format!("{errno}") }
}

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    unsupported()
}

pub struct SplitPaths<'a>(&'a !);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("unsupported")
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        *self.0
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

pub fn temp_dir() -> PathBuf {
    panic!("no standard temporary directory on this platform")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    rtabort!("exit({}) called", code);
}

pub fn getpid() -> u32 {
    panic!("no pids on this platform")
}
