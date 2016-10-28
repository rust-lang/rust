// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of `std::os` functionality for unix systems

#![allow(unused_imports)] // lots of cfg code here

use os::unix::prelude::*;

use error::Error as StdError;
use ffi::{CString, CStr, OsString, OsStr};
use fmt;
use io;
use iter;
use libc::{self, c_int, c_char, c_void};
use marker::PhantomData;
use mem;
use memchr;
use path::{self, PathBuf};
use ptr;
use slice;
use str;
use sys_common::mutex::Mutex;
use sys::cvt;
use sys::fd;
use vec;

const TMPBUF_SZ: usize = 128;
static ENV_LOCK: Mutex = Mutex::new();


extern {
    #[cfg(not(target_os = "dragonfly"))]
    #[cfg_attr(any(target_os = "linux", target_os = "emscripten", target_os = "fuchsia"),
               link_name = "__errno_location")]
    #[cfg_attr(any(target_os = "bitrig",
                   target_os = "netbsd",
                   target_os = "openbsd",
                   target_os = "android",
                   target_env = "newlib"),
               link_name = "__errno")]
    #[cfg_attr(target_os = "solaris", link_name = "___errno")]
    #[cfg_attr(any(target_os = "macos",
                   target_os = "ios",
                   target_os = "freebsd"),
               link_name = "__error")]
    #[cfg_attr(target_os = "haiku", link_name = "_errnop")]
    fn errno_location() -> *mut c_int;
}

/// Returns the platform-specific value of errno
#[cfg(not(target_os = "dragonfly"))]
pub fn errno() -> i32 {
    unsafe {
        (*errno_location()) as i32
    }
}

/// Sets the platform-specific value of errno
#[cfg(target_os = "solaris")] // only needed for readdir so far
pub fn set_errno(e: i32) {
    unsafe {
        *errno_location() = e as c_int
    }
}

#[cfg(target_os = "dragonfly")]
pub fn errno() -> i32 {
    extern {
        #[thread_local]
        static errno: c_int;
    }

    errno as i32
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errno: i32) -> String {
    extern {
        #[cfg_attr(any(target_os = "linux", target_env = "newlib"),
                   link_name = "__xpg_strerror_r")]
        fn strerror_r(errnum: c_int, buf: *mut c_char,
                      buflen: libc::size_t) -> c_int;
    }

    let mut buf = [0 as c_char; TMPBUF_SZ];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as c_int, p, buf.len()) < 0 {
            panic!("strerror_r failure");
        }

        let p = p as *const _;
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
    let p: &OsStr = p.as_ref();
    let p = CString::new(p.as_bytes())?;
    unsafe {
        match libc::chdir(p.as_ptr()) == (0 as c_int) {
            true => Ok(()),
            false => Err(io::Error::last_os_error()),
        }
    }
}

pub struct SplitPaths<'a> {
    iter: iter::Map<slice::Split<'a, u8, fn(&u8) -> bool>,
                    fn(&'a [u8]) -> PathBuf>,
}

pub fn split_paths(unparsed: &OsStr) -> SplitPaths {
    fn bytes_to_path(b: &[u8]) -> PathBuf {
        PathBuf::from(<OsStr as OsStrExt>::from_bytes(b))
    }
    fn is_colon(b: &u8) -> bool { *b == b':' }
    let unparsed = unparsed.as_bytes();
    SplitPaths {
        iter: unparsed.split(is_colon as fn(&u8) -> bool)
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
    let sep = b':';

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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "path segment contains separator `:`".fmt(f)
    }
}

impl StdError for JoinPathsError {
    fn description(&self) -> &str { "failed to join paths" }
}

pub fn current_exe() -> io::Result<PathBuf> {
    use io::ErrorKind;
    Err(io::Error::new(ErrorKind::Other, "Not yet implemented on redox"))
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

/// Returns a vector of (variable, value) byte-vector pairs for all the
/// environment variables of the current process.
pub fn env() -> Env {
    unimplemented!();
}

pub fn getenv(_k: &OsStr) -> io::Result<Option<OsString>> {
    unimplemented!();
}

pub fn setenv(_k: &OsStr, _v: &OsStr) -> io::Result<()> {
    unimplemented!();
}

pub fn unsetenv(_n: &OsStr) -> io::Result<()> {
    unimplemented!();
}

pub fn page_size() -> usize {
    4096
}

pub fn temp_dir() -> PathBuf {
    ::env::var_os("TMPDIR").map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/tmp")
    })
}

pub fn home_dir() -> Option<PathBuf> {
    return ::env::var_os("HOME").map(PathBuf::from);
}

pub fn exit(code: i32) -> ! {
    unsafe { libc::exit(code as c_int) }
}
