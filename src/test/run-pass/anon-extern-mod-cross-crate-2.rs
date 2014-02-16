// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// ignore-pretty
// aux-build:anon-extern-mod-cross-crate-1.rs
extern crate anonexternmod;

use anonexternmod::rust_get_test_int;

pub fn main() {
    unsafe {
        rust_get_test_int();
    }
}
