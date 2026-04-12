//! ThingOS PAL OS operations.
//!
//! Covers process-level OS primitives: working directory, executable path,
//! temporary directory, environment splitting, and process control.

use super::common::{
    SYS_CHDIR, SYS_CURRENT_EXE, SYS_EXIT, SYS_EXIT_GROUP, SYS_GETCWD, SYS_GETPID, cvt, syscall0,
    syscall1, syscall2,
};
use crate::error::Error as StdError;
use crate::ffi::{OsStr, OsString};
use crate::marker::PhantomData;
use crate::path::{self, PathBuf};
use crate::{fmt, io};

// ── Working directory ────────────────────────────────────────────────────────

pub fn getcwd() -> io::Result<PathBuf> {
    let mut buf = vec![0u8; 4096];
    let ret =
        unsafe { syscall2(SYS_GETCWD, buf.as_mut_ptr() as u64, buf.len() as u64) };
    let len = cvt(ret)? as usize;
    // The kernel writes a NUL-terminated string; trim the NUL.
    let trimmed = if len > 0 && buf[len - 1] == 0 { &buf[..len - 1] } else { &buf[..len] };
    let s = core::str::from_utf8(trimmed)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "getcwd returned non-UTF-8"))?;
    Ok(PathBuf::from(s))
}

pub fn chdir(path: &path::Path) -> io::Result<()> {
    let path_str = path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "non-UTF-8 path"))?;
    let bytes = path_str.as_bytes();
    let ret = unsafe { syscall2(SYS_CHDIR, bytes.as_ptr() as u64, bytes.len() as u64) };
    cvt(ret)?;
    Ok(())
}

// ── Path splitting / joining (unsupported) ───────────────────────────────────

pub struct SplitPaths<'a>(!, PhantomData<&'a ()>);

pub fn split_paths(_unparsed: &OsStr) -> SplitPaths<'_> {
    panic!("PATH splitting is not supported on ThingOS")
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
        "PATH joining is not supported on ThingOS".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "PATH joining is not supported on ThingOS"
    }
}

// ── Executable / temp / home paths ──────────────────────────────────────────

pub fn current_exe() -> io::Result<PathBuf> {
    let mut buf = vec![0u8; 4096];
    let ret =
        unsafe { syscall2(SYS_CURRENT_EXE, buf.as_mut_ptr() as u64, buf.len() as u64) };
    let len = cvt(ret)? as usize;
    let trimmed = if len > 0 && buf[len - 1] == 0 { &buf[..len - 1] } else { &buf[..len] };
    let s = core::str::from_utf8(trimmed).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, "current_exe returned non-UTF-8")
    })?;
    Ok(PathBuf::from(s))
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from("/tmp")
}

pub fn home_dir() -> Option<PathBuf> {
    // ThingOS does not expose a per-user home directory via the kernel ABI.
    None
}

// ── Process control ──────────────────────────────────────────────────────────

pub fn exit(code: i32) -> ! {
    unsafe {
        syscall1(SYS_EXIT_GROUP, code as u64);
    }
    // SYS_EXIT_GROUP must not return; fall back to SYS_EXIT just in case.
    unsafe {
        syscall1(SYS_EXIT, code as u64);
    }
    unreachable!()
}

pub fn getpid() -> u32 {
    unsafe { syscall0(SYS_GETPID) as u32 }
}
