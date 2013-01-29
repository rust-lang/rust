// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// support for test cases.

// Linkage attributes
#[ link(name = "testingfuns",
        vers = "0.6",
        uuid = "F8BF9F34-147D-47FF-9849-E6C17D34D2FF") ];

// Specify the output type
#[ crate_type = "lib" ];

#[no_core];

extern mod core(vers = "0.6");
use core::*;

extern mod std(vers = "0.6");

use core::cmp::Eq;
pub fn check_equal<T : Eq> (given : &T, expected: &T) {
    if !(given == expected) {
        fail (fmt!("given %?, expected %?",given,expected));
    }
}
