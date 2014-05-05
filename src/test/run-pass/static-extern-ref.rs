// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:static-extern-ref-2.rs

/*
 * Ensure that parse accepts references to extern static variables.
 */

extern crate bar = "static-extern-ref-2";
extern crate libc;

use libc::c_uint;

pub static test: &'static c_uint = &test_extern;

extern {
    pub static test_extern: c_uint;
}

fn main() {}
