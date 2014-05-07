// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Failure support for libcore

#![allow(dead_code)]

#[cfg(not(test))]
use str::raw::c_str_to_static_slice;

// FIXME: Once std::fmt is in libcore, all of these functions should delegate
//        to a common failure function with this signature:
//
//          extern {
//              fn rust_unwind(f: &fmt::Arguments, file: &str, line: uint) -> !;
//          }
//
//        Each of these functions can create a temporary fmt::Arguments
//        structure to pass to this function.

#[cold] #[inline(never)] // this is the slow path, always
#[lang="fail_"]
#[cfg(not(test))]
fn fail_(expr: *u8, file: *u8, line: uint) -> ! {
    unsafe {
        let expr = c_str_to_static_slice(expr as *i8);
        let file = c_str_to_static_slice(file as *i8);
        begin_unwind(expr, file, line)
    }
}

#[cold]
#[lang="fail_bounds_check"]
#[cfg(not(test))]
fn fail_bounds_check(file: *u8, line: uint, index: uint, len: uint) -> ! {
    #[allow(ctypes)]
    extern { fn rust_fail_bounds_check(file: *u8, line: uint,
                                       index: uint, len: uint,) -> !; }
    unsafe { rust_fail_bounds_check(file, line, index, len) }
}

#[cold]
pub fn begin_unwind(msg: &str, file: &'static str, line: uint) -> ! {
    #[allow(ctypes)]
    extern { fn rust_begin_unwind(msg: &str, file: &'static str,
                                  line: uint) -> !; }
    unsafe { rust_begin_unwind(msg, file, line) }
}
