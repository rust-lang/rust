use super::map_motor_error;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::motor::ffi::OsStrExt;
use crate::path::{self, PathBuf};
use crate::{fmt, io};

pub fn getcwd() -> io::Result<PathBuf> {
    moto_rt::fs::getcwd().map(PathBuf::from).map_err(map_motor_error)
}

pub fn chdir(path: &path::Path) -> io::Result<()> {
    moto_rt::fs::chdir(path.as_os_str().as_str()).map_err(map_motor_error)
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
    moto_rt::process::current_exe().map(PathBuf::from).map_err(map_motor_error)
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
    panic!("Pids on Motor OS are u64.")
}
