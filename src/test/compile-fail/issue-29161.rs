// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod a {
    struct A;

    impl Default for A {
        pub fn default() -> A {
            //~^ ERROR E0449
            A;
        }
    }
}


fn main() {
    a::A::default();
    //~^ ERROR method `default` is inaccessible
 }
