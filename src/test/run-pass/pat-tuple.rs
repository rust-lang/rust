// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(dotdot_in_tuple_patterns)]

fn b() {
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
}

fn bs() {
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
}

fn c() {
    let x = (1,);
    match x {
        (2, ..) => panic!(),
        (..) => ()
    }
}

fn cs() {
    struct S(u8);

    let x = S(1);
    match x {
        S(2, ..) => panic!(),
        S(..) => ()
    }
}

fn d() {
    let x = (1, 2, 3);
    let branch = match x {
        (1, 1, ..) => 0,
        (1, 2, 3, ..) => 1,
        (1, 2, ..) => 2,
        _ => 3
    };
    assert_eq!(branch, 1);
}

fn ds() {
    struct S(u8, u8, u8);

    let x = S(1, 2, 3);
    let branch = match x {
        S(1, 1, ..) => 0,
        S(1, 2, 3, ..) => 1,
        S(1, 2, ..) => 2,
        _ => 3
    };
    assert_eq!(branch, 1);
}

fn f() {
    let x = (1, 2, 3);
    match x {
        (1, 2, 4) => unreachable!(),
        (0, 2, 3, ..) => unreachable!(),
        (0, .., 3) => unreachable!(),
        (0, ..) => unreachable!(),
        (1, 2, 3) => (),
        (_, _, _) => unreachable!(),
    }
    match x {
        (..) => (),
    }
    match x {
        (_, _, _, ..) => (),
    }
    match x {
        (a, b, c) => {
            assert_eq!(1, a);
            assert_eq!(2, b);
            assert_eq!(3, c);
        }
    }
}

fn fs() {
    struct S(u8, u8, u8);

    let x = S(1, 2, 3);
    match x {
        S(1, 2, 4) => unreachable!(),
        S(0, 2, 3, ..) => unreachable!(),
        S(0, .., 3) => unreachable!(),
        S(0, ..) => unreachable!(),
        S(1, 2, 3) => (),
        S(_, _, _) => unreachable!(),
    }
    match x {
        S(..) => (),
    }
    match x {
        S(_, _, _, ..) => (),
    }
    match x {
        S(a, b, c) => {
            assert_eq!(1, a);
            assert_eq!(2, b);
            assert_eq!(3, c);
        }
    }
}

fn g() {
    struct S;
    struct Z;
    struct W;
    let x = (S, Z, W);
    match x { (S, ..) => {} }
    match x { (.., W) => {} }
    match x { (S, .., W) => {} }
    match x { (.., Z, _) => {} }
}

fn gs() {
    struct SS(S, Z, W);

    struct S;
    struct Z;
    struct W;
    let x = SS(S, Z, W);
    match x { SS(S, ..) => {} }
    match x { SS(.., W) => {} }
    match x { SS(S, .., W) => {} }
    match x { SS(.., Z, _) => {} }
}

fn main() {
    b();
    bs();
    c();
    cs();
    d();
    ds();
    f();
    fs();
    g();
    gs();
}
