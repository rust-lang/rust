// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

#![allow(unused)]

use m::S;

mod m {
    pub struct S(u8);

    mod n {
        use S;
        fn f() {
            S(10);
            //~^ ERROR private struct constructors are not usable through re-exports in outer modules
            //~| WARN this was previously accepted
        }
    }
}

fn main() {}
