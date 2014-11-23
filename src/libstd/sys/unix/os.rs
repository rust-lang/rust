// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc;
use libc::{c_int, c_char};
use prelude::*;
use io::IoResult;
use sys::fs::FileDesc;

use os::TMPBUF_SZ;

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
