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

use prelude::*;

use error::{FromError, Error};
use fmt;
use io::{IoError, IoResult};
use libc::{mod, c_int, c_char, c_void};
use path::BytesContainer;
use ptr;
use sync::atomic::{AtomicInt, INIT_ATOMIC_INT, SeqCst};
use sys::fs::FileDesc;
use os;

use os::TMPBUF_SZ;

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

    let mut buf = [0 as c_char, ..TMPBUF_SZ];

    let p = buf.as_mut_ptr();
    unsafe {
        if strerror_r(errno as c_int, p, buf.len() as libc::size_t) < 0 {
            panic!("strerror_r failure");
        }

        String::from_raw_buf(p as *const u8)
    }
}

pub unsafe fn pipe() -> IoResult<(FileDesc, FileDesc)> {
    let mut fds = [0, ..2];
    if libc::pipe(fds.as_mut_ptr()) == 0 {
        Ok((FileDesc::new(fds[0], true), FileDesc::new(fds[1], true)))
    } else {
        Err(super::last_error())
    }
}

pub fn getcwd() -> IoResult<Path> {
    use c_str::CString;

    let mut buf = [0 as c_char, ..BUF_BYTES];
    unsafe {
        if libc::getcwd(buf.as_mut_ptr(), buf.len() as libc::size_t).is_null() {
            Err(IoError::last_error())
        } else {
            Ok(Path::new(CString::new(buf.as_ptr(), false)))
        }
    }
}

pub unsafe fn get_env_pairs() -> Vec<Vec<u8>> {
    use c_str::CString;

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
        let env_pair =
            CString::new(*environ, false).as_bytes_no_nul().to_vec();
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

#[cfg(any(target_os = "freebsd", target_os = "dragonfly"))]
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
                         v.as_mut_ptr() as *mut c_void, &mut sz,
                         ptr::null_mut(), 0u as libc::size_t);
        if err != 0 { return None; }
        if sz == 0 { return None; }
        v.set_len(sz as uint - 1); // chop off trailing NUL
        Some(v)
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
    p.with_c_str(|buf| {
        unsafe {
            match libc::chdir(buf) == (0 as c_int) {
                true => Ok(()),
                false => Err(IoError::last_error()),
            }
        }
    })
}

pub fn page_size() -> uint {
    unsafe {
        libc::sysconf(libc::_SC_PAGESIZE) as uint
    }
}
