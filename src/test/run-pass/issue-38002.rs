// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that constant ADTs are translated OK, part k of N.

#![feature(slice_patterns)]

enum Bar {
    C
}

enum Foo {
    A {},
    B {
        y: usize,
        z: Bar
    },
}

const LIST: [(usize, Foo); 2] = [
    (51, Foo::B { y: 42, z: Bar::C }),
    (52, Foo::B { y: 45, z: Bar::C }),
];

pub fn main() {
    match LIST {
        [
            (51, Foo::B { y: 42, z: Bar::C }),
            (52, Foo::B { y: 45, z: Bar::C })
        ] => {}
        _ => {
            // I would want to print the enum here, but if
            // the discriminant is garbage this causes an
            // `unreachable` and silent process exit.
            panic!("trivial match failed")
        }
    }
}
