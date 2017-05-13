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
    y: bool,
    z: i32
}

struct Bar {
    x: i32
}

pub fn main() {
    let (x, y, z) = (1, true, 2);
    let a = Foo { x, y: y, z };
    assert_eq!(a.x, x);
    assert_eq!(a.y, y);
    assert_eq!(a.z, z);

    let b = Bar { x, };
    assert_eq!(b.x, x);

    let c = Foo { z, y, x };
    assert_eq!(c.x, x);
    assert_eq!(c.y, y);
    assert_eq!(c.z, z);
}
