// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::mem;

struct S {
    x: isize,
    y: isize,
}

type S2 = S;

struct S3<U,V> {
    x: U,
    y: V
}

type S4<U> = S3<U, char>;

fn main() {
    let s = S2 {
        x: 1,
        y: 2,
    };
    match s {
        S2 {
            x: x,
            y: y
        } => {
            assert_eq!(x, 1);
            assert_eq!(y, 2);
        }
    }
    // check that generics can be specified from the pattern
    let s = S4 {
        x: 4,
        y: 'a'
    };
    match s {
        S4::<u8> {
            x: x,
            y: y
        } => {
            assert_eq!(x, 4);
            assert_eq!(y, 'a');
            assert_eq!(mem::size_of_val(&x), 1);
        }
    };
    // check that generics can be specified from the constructor
    let s = S4::<u16> {
        x: 5,
        y: 'b'
    };
    match s {
        S4 {
            x: x,
            y: y
        } => {
            assert_eq!(x, 5);
            assert_eq!(y, 'b');
            assert_eq!(mem::size_of_val(&x), 2);
        }
    };
}
