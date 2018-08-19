// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug, PartialEq)]
struct Foo {
    x: u8,
}

pub fn main() {
    let mut foo = Foo {
        x: 1,
    };

    match &mut foo {
        Foo{x: n} => {
            *n += 1;
        },
    };

    assert_eq!(foo, Foo{x: 2});

    let Foo{x: n} = &foo;
    assert_eq!(*n, 2);
}
