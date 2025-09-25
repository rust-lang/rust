#![forbid(unsafe_op_in_unsafe_fn)]

use crate::ffi::{CStr, OsStr, OsString};
use crate::marker::PhantomData;
use crate::os::wasi::prelude::*;
use crate::path::{self, PathBuf};
use crate::sys::common::small_c_string::run_path_with_cstr;
use crate::sys::unsupported;
use crate::{fmt, io, str};

// Add a few symbols not in upstream `libc` just yet.
pub mod libc {
    pub use libc::*;

    unsafe extern "C" {
        pub fn getcwd(buf: *mut c_char, size: size_t) -> *mut c_char;
        pub fn chdir(dir: *const c_char) -> c_int;
        pub fn __wasilibc_get_environ() -> *mut *mut c_char;
    }
}

pub fn errno() -> i32 {
    unsafe extern "C" {
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
    let result = run_path_with_cstr(p, &|p| unsafe { Ok(libc::chdir(p.as_ptr())) })?;
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

impl crate::error::Error for JoinPathsError {}

pub fn current_exe() -> io::Result<PathBuf> {
    unsupported()
}

#[allow(dead_code)]
pub fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
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

pub fn cvt<T: IsMinusOne>(t: T) -> io::Result<T> {
    if t.is_minus_one() { Err(io::Error::last_os_error()) } else { Ok(t) }
}
