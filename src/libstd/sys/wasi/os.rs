use crate::any::Any;
use crate::error::Error as StdError;
use crate::ffi::{OsString, OsStr, CString, CStr};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::os::wasi::prelude::*;
use crate::path::{self, PathBuf};
use crate::ptr;
use crate::str;
use crate::sys::memchr;
use crate::sys::{cvt, unsupported, Void};
use crate::vec;

#[cfg(not(target_feature = "atomics"))]
pub unsafe fn env_lock() -> impl Any {
    // No need for a lock if we're single-threaded, but this function will need
    // to get implemented for multi-threaded scenarios
}

pub fn errno() -> i32 {
    extern {
        #[thread_local]
        static errno: libc::c_int;
    }

    unsafe { errno as i32 }
}

pub fn error_string(errno: i32) -> String {
    extern {
        fn strerror_r(errnum: libc::c_int, buf: *mut libc::c_char,
                      buflen: libc::size_t) -> libc::c_int;
    }

    let mut buf = [0 as libc::c_char; 1024];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as libc::c_int, p, buf.len()) < 0 {
            panic!("strerror_r failure");
        }
        str::from_utf8(CStr::from_ptr(p).to_bytes()).unwrap().to_owned()
    }
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
    where I: Iterator<Item=T>, T: AsRef<OsStr>
{
    Err(JoinPathsError)
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        "not supported on wasm yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str {
        "not supported on wasm yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
    _dont_send_or_sync_me: PhantomData<*mut ()>,
}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> { self.iter.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.iter.size_hint() }
}


pub fn env() -> Env {
    unsafe {
        let _guard = env_lock();
        let mut environ = libc::environ;
        let mut result = Vec::new();
        while environ != ptr::null_mut() && *environ != ptr::null_mut() {
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

    // See src/libstd/sys/unix/os.rs, same as that
    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
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
    let k = CString::new(k.as_bytes())?;
    unsafe {
        let _guard = env_lock();
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
        let _guard = env_lock();
        cvt(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(|_| ())
    }
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    let nbuf = CString::new(n.as_bytes())?;

    unsafe {
        let _guard = env_lock();
        cvt(libc::unsetenv(nbuf.as_ptr())).map(|_| ())
    }
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on wasm")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    unsafe {
        libc::exit(code)
    }
}

pub fn getpid() -> u32 {
    panic!("unsupported");
}
