// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]

// ensures that 'use foo:*' doesn't import non-public item

use m1::*;

mod foo {
    pub fn foo() {}
}
mod a {
    pub mod b {
        use foo::foo;
        type bar = int;
    }
    pub mod sub {
        use a::b::*;
        fn sub() -> bar { 1 }
        //~^ ERROR: undeclared type name
    }
}

mod m1 {
    fn foo() {}
}

fn main() {
    foo(); //~ ERROR: unresolved name
}

