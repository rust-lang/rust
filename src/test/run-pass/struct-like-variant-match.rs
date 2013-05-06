// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Foo {
    Bar {
        x: int,
        y: int
    },
    Baz {
        x: float,
        y: float
    }
}

fn f(x: &Foo) {
    match *x {
        Baz { x: x, y: y } => {
            assert!(x == 1.0);
            assert!(y == 2.0);
        }
        Bar { y: y, x: x } => {
            assert!(x == 1);
            assert!(y == 2);
        }
    }
}

pub fn main() {
    let x = Bar { x: 1, y: 2 };
    f(&x);
    let y = Baz { x: 1.0, y: 2.0 };
    f(&y);
}
