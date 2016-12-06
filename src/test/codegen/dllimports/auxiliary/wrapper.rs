// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
#![crate_type = "rlib"]

#[link(name = "dummy", kind="dylib")]
extern "C" {
    pub fn dylib_func2(x: i32) -> i32;
    pub static dylib_global2: i32;
}

#[link(name = "dummy", kind="static")]
extern "C" {
    pub fn static_func2(x: i32) -> i32;
    pub static static_global2: i32;
}
