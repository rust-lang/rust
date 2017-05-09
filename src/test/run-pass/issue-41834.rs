// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(dead_code)]

#[derive(Debug)]
struct Foo {
    one: i32,
    two: i32,
    three: i32,
}

impl Default for Foo {
    fn default() -> Self {
        Foo { one : 1, two: 2, three: 3 }
    }
}

fn main() {
    let _foo = Foo {
        one: 11,
        ..Foo::default()  // comma is optional
    };

    let _foo = Foo {
        one: 111,
        ..Foo::default(), // comma is optional
    };
}
