// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows
// ignore-macos

#![feature(macro_rules)]

extern crate native;
extern crate rustrt;
extern crate libc;
use libc::{c_char, c_int};
use native::io::process;
use rustrt::rtio;
use rustrt::c_str;

macro_rules! c_string {
    ($s:expr) => { {
        let ptr = concat!($s, "\0").as_ptr() as *const i8;
        unsafe { &c_str::CString::new(ptr, false) }
    } }
}

static EXPECTED_ERRNO: c_int = 0x778899aa;

#[no_mangle]
pub unsafe extern "C" fn chdir(_: *const c_char) -> c_int {
    // copied from std::os::errno()
    #[cfg(any(target_os = "macos",
              target_os = "ios",
              target_os = "freebsd"))]
    fn errno_location() -> *mut c_int {
        extern {
            fn __error() -> *mut c_int;
        }
        unsafe {
            __error()
        }
    }

    #[cfg(target_os = "dragonfly")]
    fn errno_location() -> *mut c_int {
        extern {
            fn __dfly_error() -> *mut c_int;
        }
        unsafe {
            __dfly_error()
        }
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn errno_location() -> *mut c_int {
        extern {
            fn __errno_location() -> *mut c_int;
        }
        unsafe {
            __errno_location()
        }
    }

    *errno_location() = EXPECTED_ERRNO;
    return -1;
}

fn main() {
    let program = c_string!("true");
    let cwd = c_string!("whatever");
    let cfg = rtio::ProcessConfig {
        program: program,
        args: &[],
        env: None,
        cwd: Some(cwd),
        stdin: rtio::Ignored,
        stdout: rtio::Ignored,
        stderr: rtio::Ignored,
        extra_io: &[],
        uid: None,
        gid: None,
        detach: false
    };

    match process::Process::spawn(cfg) {
        Ok(_) => { fail!("spawn() should have failled"); }
        Err(rtio::IoError { code: err, ..}) => {
            assert_eq!(err as c_int, EXPECTED_ERRNO);
        }
    };
}
