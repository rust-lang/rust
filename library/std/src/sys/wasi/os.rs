#![deny(unsafe_op_in_unsafe_fn)]

use crate::error::Error as StdError;
use crate::ffi::{CStr, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::marker::PhantomData;
use crate::ops::Drop;
use crate::os::wasi::prelude::*;
use crate::path::{self, PathBuf};
use crate::str;
use crate::sys::common::small_c_string::{run_path_with_cstr, run_with_cstr};
use crate::sys::memchr;
use crate::sys::unsupported;
use crate::vec;

// Add a few symbols not in upstream `libc` just yet.
mod libc {
    pub use libc::*;

    extern "C" {
        pub fn getcwd(buf: *mut c_char, size: size_t) -> *mut c_char;
        pub fn chdir(dir: *const c_char) -> c_int;
        pub fn __wasilibc_get_environ() -> *mut *mut c_char;
    }
}

cfg_if::cfg_if! {
    if #[cfg(target_feature = "atomics")] {
        // Access to the environment must be protected by a lock in multi-threaded scenarios.
        use crate::sync::{PoisonError, RwLock};
        static ENV_LOCK: RwLock<()> = RwLock::new(());
        pub fn env_read_lock() -> impl Drop {
            ENV_LOCK.read().unwrap_or_else(PoisonError::into_inner)
        }
        pub fn env_write_lock() -> impl Drop {
            ENV_LOCK.write().unwrap_or_else(PoisonError::into_inner)
        }
    } else {
        // No need for a lock if we are single-threaded.
        pub fn env_read_lock() -> impl Drop {
            Box::new(())
        }
        pub fn env_write_lock() -> impl Drop {
            Box::new(())
        }
    }
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
    let mut buf = Vec::with_capacity(512);
    loop {
        unsafe {
            let ptr = buf.as_mut_ptr() as *mut libc::c_char;
            if !libc::getcwd(ptr, buf.capacity()).is_null() {
                let len = CStr::from_ptr(buf.as_ptr() as *const libc::c_char).to_bytes().len();
                buf.set_len(len);
                buf.shrink_to_fit();
                return Ok(PathBuf::from(OsString::from_vec(buf)));
            } else {
                let error = io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::ERANGE) {
                    return Err(error);
                }
            }

            // Trigger the internal buffer resizing logic of `Vec` by requiring
            // more space than the current capacity.
            let cap = buf.capacity();
            buf.set_len(cap);
            buf.reserve(1);
        }
    }
}

pub fn chdir(p: &path::Path) -> io::Result<()> {
    let result = run_path_with_cstr(p, |p| unsafe { Ok(libc::chdir(p.as_ptr())) })?;
    match result == (0 as libc::c_int) {
        true => Ok(()),
        false => Err(io::Error::last_os_error()),
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
        "not supported on wasm yet".fmt(f)
    }
}

impl StdError for JoinPathsError {
    #[allow(deprecated)]
    fn description(&self) -> &str {
        "not supported on wasm yet"
    }
}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
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
        let _guard = env_read_lock();

        // Use `__wasilibc_get_environ` instead of `environ` here so that we
        // don't require wasi-libc to eagerly initialize the environment
        // variables.
        let mut environ = libc::__wasilibc_get_environ();

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
    let s = run_with_cstr(k.as_bytes(), |k| unsafe {
        let _guard = env_read_lock();
        Ok(libc::getenv(k.as_ptr()) as *const libc::c_char)
    })
    .ok()?;
    if s.is_null() {
        None
    } else {
        Some(OsStringExt::from_vec(unsafe { CStr::from_ptr(s) }.to_bytes().to_vec()))
    }
}

pub fn setenv(k: &OsStr, v: &OsStr) -> io::Result<()> {
    run_with_cstr(k.as_bytes(), |k| {
        run_with_cstr(v.as_bytes(), |v| unsafe {
            let _guard = env_write_lock();
            cvt(libc::setenv(k.as_ptr(), v.as_ptr(), 1)).map(drop)
        })
    })
}

pub fn unsetenv(n: &OsStr) -> io::Result<()> {
    run_with_cstr(n.as_bytes(), |nbuf| unsafe {
        let _guard = env_write_lock();
        cvt(libc::unsetenv(nbuf.as_ptr())).map(drop)
    })
}

pub fn temp_dir() -> PathBuf {
    panic!("no filesystem on wasm")
}

pub fn home_dir() -> Option<PathBuf> {
    None
}

pub fn exit(code: i32) -> ! {
    unsafe { libc::exit(code) }
}

pub fn getpid() -> u32 {
    panic!("unsupported");
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
