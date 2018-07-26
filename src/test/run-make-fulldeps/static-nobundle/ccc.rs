// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_type = "dylib"]

extern crate bbb;

pub fn do_work() {
    unsafe { bbb::native_func(); }
    bbb::wrapped_func();
}

pub fn do_work_generic<T>() {
    unsafe { bbb::native_func(); }
    bbb::wrapped_func();
}
