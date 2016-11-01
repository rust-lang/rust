// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(field_init_shorthand)]

struct Foo {
    x: i32,
    y: i32
}

fn main() {
    let (x, y, z) = (0, 1, 2);
    let foo = Foo {
        x, y, z //~ ERROR struct `Foo` has no field named `z`
    };
}

