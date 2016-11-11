// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn tuple() {
    let x = (1, 2, 3);
    match x {
        (a, b, ..) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
        }
    }
    match x {
        (.., b, c) => {
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        (a, .., c) => {
            assert_eq!(a, 1);
            assert_eq!(c, 3);
        }
    }
    match x {
        (a, b, c) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        (a, b, c, ..) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        (.., a, b, c) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
}

fn tuple_struct() {
    struct S(u8, u8, u8);

    let x = S(1, 2, 3);
    match x {
        S(a, b, ..) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
        }
    }
    match x {
        S(.., b, c) => {
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        S(a, .., c) => {
            assert_eq!(a, 1);
            assert_eq!(c, 3);
        }
    }
    match x {
        S(a, b, c) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        S(a, b, c, ..) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
    match x {
        S(.., a, b, c) => {
            assert_eq!(a, 1);
            assert_eq!(b, 2);
            assert_eq!(c, 3);
        }
    }
}

fn main() {
    tuple();
    tuple_struct();
}
