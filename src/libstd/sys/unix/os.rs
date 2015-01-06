// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of `std::os` functionality for unix systems

use prelude::v1::*;

use error::{FromError, Error};
use ffi::{self, CString};
use fmt;
use io::{IoError, IoResult};
use libc::{self, c_int, c_char, c_void};
use os::TMPBUF_SZ;
use os;
use path::{BytesContainer};
use ptr;
use str;
use sys::fs::FileDesc;

const BUF_BYTES : uint = 2048u;

/// Returns the platform-specific value of errno
pub fn errno() -> int {
    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd"))]
    fn errno_location() -> *const c_int {
        extern {
            fn __error() -> *const c_int;
        }
        unsafe {
            __error()
        }
    }

    #[cfg(target_os = "dragonfly")]
    fn errno_location() -> *const c_int {
        extern {
            fn __dfly_error() -> *const c_int;
        }
        unsafe {
            __dfly_error()
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn errno_location() -> *const c_int {
        extern {
            fn __errno_location() -> *const c_int;
        }
        unsafe {
            __errno_location()
        }
    }

    unsafe {
        (*errno_location()) as int
    }
}

/// Get a detailed string description for the given error number
pub fn error_string(errno: i32) -> String {
    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "android",
              target_os = "freebsd",
              target_os = "dragonfly"))]
    fn strerror_r(errnum: c_int, buf: *mut c_char, buflen: libc::size_t)
                  -> c_int {
        extern {
            fn strerror_r(errnum: c_int, buf: *mut c_char,
                          buflen: libc::size_t) -> c_int;
        }
        unsafe {
            strerror_r(errnum, buf, buflen)
        }
    }

    // GNU libc provides a non-compliant version of strerror_r by default
    // and requires macros to instead use the POSIX compliant variant.
    // So we just use __xpg_strerror_r which is always POSIX compliant
    #[cfg(target_os = "linux")]
    fn strerror_r(errnum: c_int, buf: *mut c_char,
                  buflen: libc::size_t) -> c_int {
        extern {
            fn __xpg_strerror_r(errnum: c_int,
                                buf: *mut c_char,
                                buflen: libc::size_t)
                                -> c_int;
        }
        unsafe {
            __xpg_strerror_r(errnum, buf, buflen)
        }
    }

    let mut buf = [0 as c_char; TMPBUF_SZ];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as c_int, p, buf.len() as libc::size_t) < 0 {
            panic!("strerror_r failure");
        }

        let p = p as *const _;
        str::from_utf8(ffi::c_str_to_bytes(&p)).unwrap().to_string()
    }
}

pub unsafe fn pipe() -> IoResult<(FileDesc, FileDesc)> {
    let mut fds = [0; 2];
    if libc::pipe(fds.as_mut_ptr()) == 0 {
        Ok((FileDesc::new(fds[0], true), FileDesc::new(fds[1], true)))
    } else {
        Err(super::last_error())
    }
}

pub fn getcwd() -> IoResult<Path> {
    let mut buf = [0 as c_char; BUF_BYTES];
    unsafe {
        if libc::getcwd(buf.as_mut_ptr(), buf.len() as libc::size_t).is_null() {
            Err(IoError::last_error())
        } else {
            Ok(Path::new(ffi::c_str_to_bytes(&buf.as_ptr())))
        }
    }
}

pub unsafe fn get_env_pairs() -> Vec<Vec<u8>> {
    extern {
        fn rust_env_pairs() -> *const *const c_char;
    }
    let mut environ = rust_env_pairs();
    if environ as uint == 0 {
        panic!("os::env() failure getting env string from OS: {}",
               os::last_os_error());
    }
    let mut result = Vec::new();
    while *environ != 0 as *const _ {
        let env_pair = ffi::c_str_to_bytes(&*environ).to_vec();
        result.push(env_pair);
        environ = environ.offset(1);
    }
    result
}

pub fn split_paths(unparsed: &[u8]) -> Vec<Path> {
    unparsed.split(|b| *b == b':').map(Path::new).collect()
}

pub fn join_paths<T: BytesContainer>(paths: &[T]) -> Result<Vec<u8>, &'static str> {
    let mut joined = Vec::new();
    let sep = b':';

    for (i, path) in paths.iter().map(|p| p.container_as_bytes()).enumerate() {
        if i > 0 { joined.push(sep) }
        if path.contains(&sep) { return Err("path segment contains separator `:`") }
        joined.push_all(path);
    }

    Ok(joined)
}

#[cfg(target_os = "freebsd")]
pub fn load_self() -> Option<Vec<u8>> {
    unsafe {
        use libc::funcs::bsd44::*;
        use libc::consts::os::extra::*;
        let mut mib = vec![CTL_KERN as c_int,
                           KERN_PROC as c_int,
                           KERN_PROC_PATHNAME as c_int,
                           -1 as c_int];
        let mut sz: libc::size_t = 0;
        let err = sysctl(mib.as_mut_ptr(), mib.len() as ::libc::c_uint,
                         ptr::null_mut(), &mut sz, ptr::null_mut(),
                         0u as libc::size_t);
        if err != 0 { return None; }
        if sz == 0 { return None; }
        let mut v: Vec<u8> = Vec::with_capacity(sz as uint);
        let err = sysctl(mib.as_mut_ptr(), mib.len() as ::libc::c_uint,
                         v.as_mut_ptr() as *mut libc::c_void, &mut sz,
                         ptr::null_mut(), 0u as libc::size_t);
        if err != 0 { return None; }
        if sz == 0 { return None; }
        v.set_len(sz as uint - 1); // chop off trailing NUL
        Some(v)
    }
}

#[cfg(target_os = "dragonfly")]
pub fn load_self() -> Option<Vec<u8>> {
    use std::io;

    match io::fs::readlink(&Path::new("/proc/curproc/file")) {
        Ok(path) => Some(path.into_vec()),
        Err(..) => None
    }
}

#[cfg(any(target_os = "linux", target_os = "android"))]
pub fn load_self() -> Option<Vec<u8>> {
    use std::io;

    match io::fs::readlink(&Path::new("/proc/self/exe")) {
        Ok(path) => Some(path.into_vec()),
        Err(..) => None
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub fn load_self() -> Option<Vec<u8>> {
    unsafe {
        use libc::funcs::extra::_NSGetExecutablePath;
        let mut sz: u32 = 0;
        _NSGetExecutablePath(ptr::null_mut(), &mut sz);
        if sz == 0 { return None; }
        let mut v: Vec<u8> = Vec::with_capacity(sz as uint);
        let err = _NSGetExecutablePath(v.as_mut_ptr() as *mut i8, &mut sz);
        if err != 0 { return None; }
        v.set_len(sz as uint - 1); // chop off trailing NUL
        Some(v)
    }
}

pub fn chdir(p: &Path) -> IoResult<()> {
    let p = CString::from_slice(p.as_vec());
    unsafe {
        match libc::chdir(p.as_ptr()) == (0 as c_int) {
            true => Ok(()),
            false => Err(IoError::last_error()),
        }
    }
}

pub fn page_size() -> uint {
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as uint
    }
}
