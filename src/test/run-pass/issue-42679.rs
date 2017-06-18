// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
#![feature(box_patterns)]

#[derive(Debug, PartialEq)]
enum Test {
    Foo(usize),
    Bar(isize),
}

fn main() {
    let a = box Test::Foo(10);
    let b = box Test::Bar(-20);
    match (a, b) {
        (_, box Test::Foo(_)) => unreachable!(),
        (box Test::Foo(x), b) => {
            assert_eq!(x, 10);
            assert_eq!(b, box Test::Bar(-20));
        },
        _ => unreachable!(),
    }
}
