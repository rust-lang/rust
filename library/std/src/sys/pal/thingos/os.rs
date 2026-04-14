//! ThingOS PAL — OS-level syscall bindings.
//!
//! All syscall numbers come from the shared ABI definitions (abi/src/numbers.rs).
//! Unsupported operations return `io::Error::UNSUPPORTED_PLATFORM` explicitly.

use super::raw_syscall6;
use crate::ffi::{OsStr, OsString};
use crate::path::{self, PathBuf};
use crate::{fmt, io};

// Syscall numbers (from abi/src/numbers.rs)
const SYS_EXIT: u32 = 0x1000;
const SYS_GETPID: u32 = 0x1002;
const SYS_FS_CHDIR: u32 = 0x4012;
const SYS_FS_GETCWD: u32 = 0x4013;
const SYS_FS_READLINK: u32 = 0x401A;

/// Converts a negative syscall return value to an io::Error.
#[inline]
fn syscall_err(ret: isize) -> io::Error {
    io::Error::from_raw_os_error((-ret) as i32)
}

pub fn getcwd() -> io::Result<PathBuf> {
    // First call: buf_ptr=0 returns the number of bytes needed.
    let needed = unsafe { raw_syscall6(SYS_FS_GETCWD, 0, 0, 0, 0, 0, 0) };
    if needed < 0 {
        return Err(syscall_err(needed));
    }
    let needed = needed as usize;
    if needed == 0 {
        return Ok(PathBuf::from("/"));
    }
    let mut buf = crate::vec![0u8; needed];
    let ret = unsafe {
        raw_syscall6(SYS_FS_GETCWD, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0, 0)
    };
    if ret < 0 {
        return Err(syscall_err(ret));
    }
    let n = ret as usize;
    buf.truncate(n);
    let s = crate::string::String::from_utf8(buf)
        .map_err(|_| io::const_error!(io::ErrorKind::InvalidData, "cwd is not valid UTF-8"))?;
    Ok(PathBuf::from(s))
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let path = p
        .to_str()
        .ok_or_else(|| io::const_error!(io::ErrorKind::InvalidInput, "path is not valid UTF-8"))?;
    let ret = unsafe {
        raw_syscall6(SYS_FS_CHDIR, path.as_ptr() as usize, path.len(), 0, 0, 0, 0)
    };
    if ret < 0 { Err(syscall_err(ret)) } else { Ok(()) }
}

pub struct SplitPaths<'a> {
    iter: crate::slice::Split<'a, u8, fn(&u8) -> bool>,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    fn is_colon(b: &u8) -> bool {
        *b == b':'
    }
    SplitPaths { iter: unparsed.as_encoded_bytes().split(is_colon as fn(&u8) -> bool) }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.iter.next().map(|b| {
            // SAFETY: bytes came from an OsStr via as_encoded_bytes, so they are valid.
            let s = unsafe { OsStr::from_encoded_bytes_unchecked(b) };
            PathBuf::from(s)
        })
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let mut joined = crate::vec::Vec::new();
    for (i, path) in paths.enumerate() {
        let bytes = path.as_ref().as_encoded_bytes();
        if bytes.contains(&b':') {
            return Err(JoinPathsError);
        }
        if i > 0 {
            joined.push(b':');
        }
        joined.extend_from_slice(bytes);
    }
    // SAFETY: bytes were assembled from OsStr encoded bytes, which are always valid.
    Ok(unsafe { OsString::from_encoded_bytes_unchecked(joined) })
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "path segment contains separator `:`".fmt(f)
    }
}

impl crate::error::Error for JoinPathsError {}

pub fn exit(code: i32) -> ! {
    unsafe { raw_syscall6(SYS_EXIT, code as usize, 0, 0, 0, 0, 0) };
    loop {}
}

pub fn getpid() -> u32 {
    let ret = unsafe { raw_syscall6(SYS_GETPID, 0, 0, 0, 0, 0, 0) };
    if ret < 0 { 0 } else { ret as u32 }
}

pub fn home_dir() -> Option<PathBuf> {
    // ThingOS doesn't have a traditional system user DB.  Return None;
    // callers fall back to querying the HOME env var.
    None
}

pub fn temp_dir() -> PathBuf {
    PathBuf::from("/tmp")
}

/// Returns the path of the current process's executable by reading the
/// `/proc/self/exe` symlink exposed by the ThingOS kernel.
pub fn current_exe() -> io::Result<PathBuf> {
    const PATH: &[u8] = b"/proc/self/exe";
    let mut buf = crate::vec![0u8; 4096];
    let ret = unsafe {
        raw_syscall6(
            SYS_FS_READLINK,
            PATH.as_ptr() as usize,
            PATH.len(),
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
        )
    };
    if ret < 0 {
        return Err(syscall_err(ret));
    }
    let n = ret as usize;
    buf.truncate(n);
    let s = crate::string::String::from_utf8(buf).map_err(|_| {
        io::const_error!(io::ErrorKind::InvalidData, "current_exe path is not valid UTF-8")
    })?;
    Ok(PathBuf::from(s))
}
