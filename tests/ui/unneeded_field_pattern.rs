// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![warn(clippy::unneeded_field_pattern)]
#[allow(dead_code, unused)]

struct Foo {
    a: i32,
    b: i32,
    c: i32,
}

fn main() {
    let f = Foo { a: 0, b: 0, c: 0 };

    match f {
        Foo { a: _, b: 0, .. } => {},

        Foo { a: _, b: _, c: _ } => {},
    }
    match f {
        Foo { b: 0, .. } => {}, // should be OK
        Foo { .. } => {},       // and the Force might be with this one
    }
}
