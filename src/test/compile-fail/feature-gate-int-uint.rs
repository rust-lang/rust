// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

mod u {
    type X = uint; //~ WARN the `uint` type is deprecated
    struct Foo {
        x: uint //~ WARN the `uint` type is deprecated
    }
    fn bar(x: uint) { //~ WARN the `uint` type is deprecated
        1u; //~ WARN the `u` suffix on integers is deprecated
    }
}
mod i {
    type X = int; //~ WARN the `int` type is deprecated
    struct Foo {
        x: int //~ WARN the `int` type is deprecated
    }
    fn bar(x: int) { //~ WARN the `int` type is deprecated
        1i; //~ WARN the `i` suffix on integers is deprecated
    }
}

fn main() {
    // make compilation fail, after feature gating
    let () = 1u8; //~ ERROR
}
