//! Implementation of `std::os` functionality for unix systems

#![allow(unused_imports)] // lots of cfg code here

use libc::c_char;

use crate::os::unix::prelude::*;

use crate::error::Error as StdError;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io::{self, Read, Write};
use crate::iter;
use crate::marker::PhantomData;
use crate::mem;
use crate::memchr;
use crate::path::{self, PathBuf};
use crate::ptr;
use crate::slice;
use crate::str;
use crate::sys_common::mutex::Mutex;
use crate::sys::{cvt, cvt_libc, fd, syscall};
use crate::vec;

extern {
    #[link_name = "__errno_location"]
    fn errno_location() -> *mut i32;
}

/// Returns the platform-specific value of errno
pub fn errno() -> i32 {
    unsafe {
        (*errno_location())
    }
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errno: i32) -> String {
    if let Some(string) = syscall::STR_ERROR.get(errno as usize) {
        string.to_string()
    } else {
        "unknown error".to_string()
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    let mut buf = [0; 4096];
    let count = cvt(syscall::getcwd(&mut buf))?;
    Ok(PathBuf::from(OsString::from_vec(buf[.. count].to_vec())))
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    cvt(syscall::chdir(p.to_str().unwrap())).and(Ok(()))
}

pub struct SplitPaths<'a> {
    iter: iter::Map<slice::Split<'a, u8, fn(&u8) -> bool>,
                    fn(&'a [u8]) -> PathBuf>,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    fn bytes_to_path(b: &[u8]) -> PathBuf {
        PathBuf::from(<OsStr as OsStrExt>::from_bytes(b))
    }
    fn is_semicolon(b: &u8) -> bool { *b == b';' }
    let unparsed = unparsed.as_bytes();
    SplitPaths {
        iter: unparsed.split(is_semicolon as fn(&u8) -> bool)
                      .map(bytes_to_path as fn(&[u8]) -> PathBuf)
    }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
    where I: Iterator<Item=T>, T: AsRef<OsStr>
{
    let mut joined = Vec::new();
    let sep = b';';

    for (i, path) in paths.enumerate() {
        let path = path.as_ref().as_bytes();
        if i > 0 { joined.push(sep) }
        if path.contains(&sep) {
            return Err(JoinPathsError)
        }
        joined.extend_from_slice(path);
    }
    Ok(OsStringExt::from_vec(joined))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "path segment contains separator `:`".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str { "failed to join paths" }
}

pub fn current_exe() -> io::Result<PathBuf> {
    use crate::fs::File;

    let mut file = File::open("sys:exe")?;

    let mut path = String::new();
    file.read_to_string(&mut path)?;

    if path.ends_with('\n') {
        path.pop();
    }

    Ok(PathBuf::from(path))
}

pub static ENV_LOCK: Mutex = Mutex::new();

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}

pub unsafe fn environ() -> *mut *const *const c_char {
    extern { static mut environ: *const *const c_char; }
    &mut environ
}

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    unsafe {
        let _guard = ENV_LOCK.lock();
        let mut environ = *environ();
        if environ == ptr::null() {
            panic!("os::env() failure getting env string from OS: {}",
                   io::Error::last_os_error());
        }
        let mut result = Vec::new();
        while *environ != ptr::null() {
            if let Some(key_value) = parse(CStr::from_ptr(*environ).to_bytes()) {
                result.push(key_value);
            }
            environ = environ.offset(1);
        }
        return Env {
            iter: result.into_iter(),
            _dont_send_or_sync_me: PhantomData,
        }
    }

    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
        // Strategy (copied from glibc): Variable name and value are separated
        // by an ASCII equals sign '='. Since a variable name must not be
        // empty, allow variable names starting with an equals sign. Skip all
        // malformed lines.
        if input.is_empty() {
            return None;
        }
        let pos = memchr::memchr(b'=', &input[1..]).map(|p| p + 1);
        pos.map(|p| (
            OsStringExt::from_vec(input[..p].to_vec()),
            OsStringExt::from_vec(input[p+1..].to_vec()),
        ))
    }
}

pub fn getenv(k: &OsStr) -> io::Result<Option<OsString>> {
    // environment variables with a nul byte can't be set, so their value is
    // always None as well
    let k = CString::new(k.as_bytes())?;
    unsafe {
        let _guard = ENV_LOCK.lock();
        let s = libc::getenv(k.as_ptr()) as *const libc::c_char;
        let ret = if s.is_null() {
            None
        } else {
            Some(OsStringExt::from_vec(CStr::from_ptr(s).to_bytes().to_vec()))
        };
        Ok(ret)
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let k = CString::new(k.as_bytes())?;
    let v = CString::new(v.as_bytes())?;

    unsafe {
        let _guard = ENV_LOCK.lock();
        cvt_libc(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(|_| ())
    }
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    let nbuf = CString::new(n.as_bytes())?;

    unsafe {
        let _guard = ENV_LOCK.lock();
        cvt_libc(libc::unsetenv(nbuf.as_ptr())).map(|_| ())
    }
}

pub fn page_size() -> usize {
    4096
}

pub fn temp_dir() -> PathBuf {
    crate::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/tmp")
    })
}

pub fn home_dir() -> Option<PathBuf> {
    return crate::env::var_os("HOME").map(PathBuf::from);
}

pub fn exit(code: i32) -> ! {
    let _ = syscall::exit(code as usize);
    unreachable!();
}

pub fn getpid() -> u32 {
    syscall::getpid().unwrap() as u32
}

pub fn getppid() -> u32 {
    syscall::getppid().unwrap() as u32
}
