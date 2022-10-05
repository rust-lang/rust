#![deny(unsafe_op_in_unsafe_fn)]

use crate::sync::{Mutex, MutexGuard};
use crate::error::Error as StdError;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::os::wasi::prelude::*;
use crate::iter;
use crate::slice;
use crate::path::{self, PathBuf};
use crate::str;
use crate::sys::memchr;
use crate::vec;
use crate::lazy::SyncLazy;

use super::err2io;

const PATH_SEPARATOR: u8 = b':';

static ENV_LOCK: SyncLazy<Mutex<()>> = SyncLazy::new(|| Mutex::new(()));

pub fn env_lock<'a>() -> MutexGuard<'a, ()> {
    ENV_LOCK.lock().unwrap()
}

pub fn errno() -> i32 {
    extern "C" {
        #[thread_local]
        static errno: libc::c_int;
    }

    unsafe { errno as i32 }
}

pub fn error_string(errno: i32) -> String {
    let mut buf = [0 as libc::c_char; 1024];

    let p = buf.as_mut_ptr();
    unsafe {
        if libc::strerror_r(errno as libc::c_int, p, buf.len()) < 0 {
            panic!("strerror_r failure");
        }
        str::from_utf8(CStr::from_ptr(p).to_bytes()).unwrap().to_owned()
    }
}

pub fn getcwd() -> io::Result<PathBuf> {
    let mut buf = Vec::<u8>::with_capacity(1024);
    for _ in 0..2 {
        unsafe {
            let mut len = buf.capacity() as usize;
            let ptr_buf = buf.as_mut_ptr() as *mut u8;
            let ptr_len = &mut len as *mut usize;
            match wasi::getcwd(ptr_buf, ptr_len) {
                Ok(()) => {
                    drop(ptr_len);
                    drop(ptr_buf);
                    buf.set_len(len as usize);
                    buf.shrink_to_fit();
                    return Ok(PathBuf::from(OsString::from_vec(buf)));
                },
                Err(wasi::ERRNO_OVERFLOW) => {
                    buf = Vec::with_capacity(len as usize);
                    continue;
                },
                Err(err) => {
                    return Err(err2io(err));
                }
            }
        }
    }
    Err(err2io(wasi::ERRNO_INVAL))
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let p = p.to_str()
        .ok_or_else(|| err2io(wasi::ERRNO_INVAL))?;
    unsafe {
        wasi::chdir(p).map_err(err2io)
    }
}

pub struct SplitPaths<'a> {
    iter: iter::Map<slice::Split<'a, u8, fn(&u8) -> bool>, fn(&'a [u8]) -> PathBuf>,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths<'_> {
    fn bytes_to_path(b: &[u8]) -> PathBuf {
        PathBuf::from(<OsStr as OsStrExt>::from_bytes(b))
    }
    fn is_separator(b: &u8) -> bool {
        *b == PATH_SEPARATOR
    }
    let unparsed = unparsed.as_bytes();
    SplitPaths {
        iter: unparsed
            .split(is_separator as fn(&u8) -> bool)
            .map(bytes_to_path as fn(&[u8]) -> PathBuf),
    }
}

impl<'a> Iterator for SplitPaths<'a> {
    type Item = PathBuf;
    fn next(&mut self) -> Option<PathBuf> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

#[derive(Debug)]
pub struct JoinPathsError;

pub fn join_paths<I, T>(paths: I) -> Result<OsString, JoinPathsError>
where
    I: Iterator<Item = T>,
    T: AsRef<OsStr>,
{
    let mut joined = Vec::new();

    for (i, path) in paths.enumerate() {
        let path = path.as_ref().as_bytes();
        if i > 0 {
            joined.push(PATH_SEPARATOR)
        }
        if path.contains(&PATH_SEPARATOR) {
            return Err(JoinPathsError);
        }
        joined.extend_from_slice(path);
    }
    Ok(OsStringExt::from_vec(joined))
}

impl fmt::Display for JoinPathsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "path segment contains separator `{}`", char::from(PATH_SEPARATOR))
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "failed to join paths"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    use crate::io::ErrorKind;
    Err(io::const_io_error!(ErrorKind::Unsupported, "Not yet implemented!"))
}

pub struct Env {
    iter: vec::IntoIter<(OsString, OsString)>,
}

impl !Send for Env {}
impl !Sync for Env {}

impl Iterator for Env {
    type Item = (OsString, OsString);
    fn next(&mut self) -> Option<(OsString, OsString)> {
        self.iter.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub fn env() -> Env {
    unsafe {
        let _guard = env_lock();
        let mut environ = libc::environ;
        let mut result = Vec::new();
        if !environ.is_null() {
            while !(*environ).is_null() {
                if let Some(key_value) = parse(CStr::from_ptr(*environ).to_bytes()) {
                    result.push(key_value);
                }
                environ = environ.add(1);
            }
        }
        return Env { iter: result.into_iter() };
    }

    // See src/libstd/sys/unix/os.rs, same as that
    fn parse(input: &[u8]) -> Option<(OsString, OsString)> {
        if input.is_empty() {
            return None;
        }
        let pos = memchr::memchr(b'=', &input[1..]).map(|p| p + 1);
        pos.map(|p| {
            (
                OsStringExt::from_vec(input[..p].to_vec()),
                OsStringExt::from_vec(input[p + 1..].to_vec()),
            )
        })
    }
}

pub fn getenv(k: &OsStr) -> Option<OsString> {
    let k = CString::new(k.as_bytes()).ok()?;
    unsafe {
        let _guard = env_lock();
        let s = libc::getenv(k.as_ptr()) as *const libc::c_char;
        if s.is_null() {
            None
        } else {
            Some(OsStringExt::from_vec(CStr::from_ptr(s).to_bytes().to_vec()))
        }
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    let k = CString::new(k.as_bytes())?;
    let v = CString::new(v.as_bytes())?;

    unsafe {
        let _guard = env_lock();
        cvt(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(drop)
    }
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    let nbuf = CString::new(n.as_bytes())?;

    unsafe {
        let _guard = env_lock();
        cvt(libc::unsetenv(nbuf.as_ptr())).map(drop)
    }
}

#[allow(dead_code)]
pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

pub fn temp_dir() -> PathBuf {
    crate::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/tmp")
    })
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    unsafe { libc::exit(code) }
}

pub fn getpid() -> u32 {
    unsafe {
        wasi::proc_id()
            .map(|a| a as u32)
            .map_err(err2io)
            .unwrap_or_else(|_| 0u32)
    }
}

#[doc(hidden)]
pub trait IsMinusOne {
    fn is_minus_one(&self) -> bool;
}

macro_rules! impl_is_minus_one {
    ($($t:ident)*) => ($(impl IsMinusOne for $t {
        fn is_minus_one(&self) -> bool {
            *self == -1
        }
    })*)
}

impl_is_minus_one! { i8 i16 i32 i64 isize }

fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(io::Error::last_os_error()) } else { Ok(t) }
}