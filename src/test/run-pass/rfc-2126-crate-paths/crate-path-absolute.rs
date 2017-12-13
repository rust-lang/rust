// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(crate_in_paths)]

use crate::m::f;
use crate as root;

mod m {
    pub fn f() -> u8 { 1 }
    pub fn g() -> u8 { 2 }
    pub fn h() -> u8 { 3 }

    // OK, visibilities are implicitly absolute like imports
    pub(in crate::m) struct S;
}

mod n
{
    use crate::m::f;
    use crate as root;
    pub fn check() {
        assert_eq!(f(), 1);
        assert_eq!(::crate::m::g(), 2);
        assert_eq!(root::m::h(), 3);
    }
}

fn main() {
    assert_eq!(f(), 1);
    assert_eq!(::crate::m::g(), 2);
    assert_eq!(root::m::h(), 3);
    n::check();
}
