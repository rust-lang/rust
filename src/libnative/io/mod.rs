// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Native thread-blocking I/O implementation
//!
//! This module contains the implementation of native thread-blocking
//! implementations of I/O on all platforms. This module is not intended to be
//! used directly, but rather the rust runtime will fall back to using it if
//! necessary.
//!
//! Rust code normally runs inside of green tasks with a local scheduler using
//! asynchronous I/O to cooperate among tasks. This model is not always
//! available, however, and that's where these native implementations come into
//! play. The only dependencies of these modules are the normal system libraries
//! that you would find on the respective platform.

#![allow(non_snake_case)]

use libc::{mod, c_int};
use std::c_str::CString;
use std::os;
use std::rt::rtio::{mod, IoResult, IoError};
use std::num;

#[cfg(windows)]
#[path = "tty_windows.rs"]
mod tty;

fn unimpl() -> IoError {
    #[cfg(unix)] use libc::ENOSYS as ERROR;
    #[cfg(windows)] use libc::ERROR_CALL_NOT_IMPLEMENTED as ERROR;
    IoError {
        code: ERROR as uint,
        extra: 0,
        detail: Some("not yet supported by the `native` runtime, maybe try `green`.".to_string()),
    }
}

fn last_error() -> IoError {
    let errno = os::errno() as uint;
    IoError {
        code: os::errno() as uint,
        extra: 0,
        detail: Some(os::error_string(errno)),
    }
}

#[cfg(windows)]
#[inline]
fn retry<I> (f: || -> I) -> I { f() } // PR rust-lang/rust/#17020

#[cfg(unix)]
#[inline]
fn retry<I: PartialEq + num::One + Neg<I>> (f: || -> I) -> I {
    let minus_one = -num::one::<I>();
    loop {
        let n = f();
        if n == minus_one && os::errno() == libc::EINTR as int { }
        else { return n }
    }
}


fn keep_going(data: &[u8], f: |*const u8, uint| -> i64) -> i64 {
    let origamt = data.len();
    let mut data = data.as_ptr();
    let mut amt = origamt;
    while amt > 0 {
        let ret = retry(|| f(data, amt));
        if ret == 0 {
            break
        } else if ret != -1 {
            amt -= ret as uint;
            data = unsafe { data.offset(ret as int) };
        } else {
            return ret;
        }
    }
    return (origamt - amt) as i64;
}

/// Implementation of rt::rtio's IoFactory trait to generate handles to the
/// native I/O functionality.
pub struct IoFactory {
    _cannot_construct_outside_of_this_module: ()
}

impl IoFactory {
    pub fn new() -> IoFactory {
        IoFactory { _cannot_construct_outside_of_this_module: () }
    }
}

impl rtio::IoFactory for IoFactory {
}
