// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


struct Triple {a: int, b: int, c: int}

fn test(foo: Box<Triple>) -> Box<Triple> {
    let foo = foo;
    let bar = foo;
    let baz = bar;
    let quux = baz;
    return quux;
}

pub fn main() {
    let x = box Triple{a: 1, b: 2, c: 3};
    let y = test(x);
    assert!((y.c == 3));
}
