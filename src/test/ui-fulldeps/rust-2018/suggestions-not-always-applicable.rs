// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:suggestions-not-always-applicable.rs
// edition:2015
// run-rustfix
// rustfix-only-machine-applicable
// compile-pass

#![feature(rust_2018_preview)]
#![warn(rust_2018_compatibility)]

extern crate suggestions_not_always_applicable as foo;

pub struct Foo;

mod test {
    use crate::foo::foo;

    #[foo] //~ WARN: absolute paths must start with
    //~| WARN: previously accepted
    //~| WARN: absolute paths
    //~| WARN: previously accepted
    fn main() {
    }
}

fn main() {
    test::foo();
}
