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
use core::cmp;

pub pure fn check_equal_ptr<T : cmp::Eq> (given : &T, expected: &T) {
    if !((given == expected) && (expected == given )) {
        die!(fmt!("given %?, expected %?",given,expected));
    }
}

pub pure fn check_equal<T : cmp::Eq> (given : T, expected: T) {
    if !((given == expected) && (expected == given )) {
        die!(fmt!("given %?, expected %?",given,expected));
    }
}
