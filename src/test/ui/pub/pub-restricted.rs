// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(pub_restricted)]

mod a {}

pub (a) fn afn() {} //~ incorrect visibility restriction
pub (b) fn bfn() {} //~ incorrect visibility restriction
pub fn privfn() {}
mod x {
    mod y {
        pub (in x) fn foo() {}
        pub (super) fn bar() {}
        pub (crate) fn qux() {}
    }
}

mod y {
    struct Foo {
        pub (crate) c: usize,
        pub (super) s: usize,
        valid_private: usize,
        pub (in y) valid_in_x: usize,
        pub (a) invalid: usize, //~ incorrect visibility restriction
        pub (in x) non_parent_invalid: usize, //~ ERROR visibilities can only be restricted
    }
}

fn main() {}

// test multichar names
mod xyz {}
pub (xyz) fn xyz() {} //~ incorrect visibility restriction
