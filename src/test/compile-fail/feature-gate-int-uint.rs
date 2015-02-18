// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code, unused_variables)]
#![feature(rustc_attrs)]

mod u {
    type X = uint; //~ WARN the `uint` type is deprecated
    struct Foo {
        x: uint //~ WARN the `uint` type is deprecated
    }
    fn bar(x: uint) { //~ WARN the `uint` type is deprecated
        1_u; //~ WARN the `u` and `us` suffixes on integers are deprecated
        1_us; //~ WARN the `u` and `us` suffixes on integers are deprecated
    }
}
mod i {
    type X = int; //~ WARN the `int` type is deprecated
    struct Foo {
        x: int //~ WARN the `int` type is deprecated
    }
    fn bar(x: int) { //~ WARN the `int` type is deprecated
        1_i; //~ WARN the `i` and `is` suffixes on integers are deprecated
        1_is; //~ WARN the `i` and `is` suffixes on integers are deprecated
    }
}

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
