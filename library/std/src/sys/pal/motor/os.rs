use super::map_motor_error;
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::motor::ffi::OsStrExt;
use crate::path::{self, PathBuf};
use crate::sys::RawOsError;
use crate::{fmt, io};

pub fn errno() -> RawOsError {
    // Not used in Motor OS because it is ambiguous: Motor OS
    // is micro-kernel-based, and I/O happens via a shared-memory
    // ring buffer, so an I/O operation that on a unix is a syscall
    // may involve no sycalls on Motor OS at all, or a syscall
    // that e.g. waits for a notification from the I/O driver
    // (sys-io); and the wait syscall may succeed, but the
    // driver may report an I/O error; or a bunch of results
    // for several I/O operations, some successful and some
    // not.
    //
    // Also I/O operations in a Motor OS process are handled by a
    // separate runtime background/I/O thread, so it is really hard
    // to define what "last system error in the current thread"
    // actually means.
    moto_rt::E_UNKNOWN.into()
}

pub fn error_string(errno: RawOsError) -> String {
    let error_code: moto_rt::ErrorCode = match errno {
        x if x < 0 => moto_rt::E_UNKNOWN,
        x if x > u16::MAX.into() => moto_rt::E_UNKNOWN,
        x => x as moto_rt::ErrorCode, /* u16 */
    };
    format!("{}", moto_rt::Error::from(error_code))
}

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
