// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ffi::CStr;
use libc::{self, c_int};
use str;

pub use sys::cloudabi::shims::os::*;

pub fn errno() -> i32 {
    extern "C" {
        #[thread_local]
        static errno: c_int;
    }

    unsafe { errno as i32 }
}

/// Gets a detailed string description for the given error number.
pub fn error_string(errno: i32) -> String {
    // cloudlibc's strerror() is guaranteed to be thread-safe. There is
    // thus no need to use strerror_r().
    str::from_utf8(unsafe { CStr::from_ptr(libc::strerror(errno)) }.to_bytes())
        .unwrap()
        .to_owned()
}

pub fn exit(code: i32) -> ! {
    unsafe { libc::exit(code as c_int) }
}
