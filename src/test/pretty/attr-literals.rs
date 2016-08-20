// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pp-exact
// Tests literals in attributes.

#![feature(custom_attribute, attr_literals)]

fn main() {
    #![hello("hi", 1, 2, 1.012, pi = 3.14, bye, name("John"))]
    #[align = 8]
    fn f() { }

    #[vec(1, 2, 3)]
    fn g() { }
}
