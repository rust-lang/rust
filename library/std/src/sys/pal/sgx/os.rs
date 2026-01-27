use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::path::{self, PathBuf};
use crate::sys::{sgx_ineffective, unsupported};
use crate::{fmt, io};

pub fn getcwd() -> io::Result<PathBuf> {
    unsupported()
}

pub fn chdir(_: &path::Path) -> io::Result<()> {
    sgx_ineffective(())
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
        "not supported in SGX yet".fmt(f)
    }
}

impl crate::error::Error for JoinPathsError {}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem in SGX")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    super::abi::exit_with_code(code as _)
}

pub fn getpid() -> u32 {
    panic!("no pids in SGX")
}
