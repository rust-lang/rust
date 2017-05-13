// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(pub_restricted)]
#![deny(unused)]

mod foo {
    fn f() {}

    mod m1 {
        pub(super) use super::f; //~ ERROR unused
    }

    mod m2 {
        #[allow(unused)]
        use super::m1::*; // (despite this glob import)
    }

    mod m3 {
        pub(super) use super::f; // Check that this is counted as used (c.f. #36249).
    }

    pub mod m4 {
        use super::m3::*;
        pub fn g() { f(); }
    }
}

fn main() {
    foo::m4::g();
}
