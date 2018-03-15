// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const A: i32 = Foo::B; //~ ERROR E0265
                       //~^ NOTE recursion not allowed in constant

enum Foo {
    B = A, //~ ERROR E0265
           //~^ NOTE recursion not allowed in constant
}

enum Bar {
    C = Bar::C, //~ ERROR E0265
                //~^ NOTE recursion not allowed in constant
}

const D: i32 = A;

fn main() {}
