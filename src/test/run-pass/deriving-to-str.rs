// xfail-fast #6330
// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rand;

#[deriving(Rand,ToStr)]
struct A;

#[deriving(Rand,ToStr)]
struct B(int, int);

#[deriving(Rand,ToStr)]
struct C {
    x: f64,
    y: (u8, u8)
}

#[deriving(Rand,ToStr)]
enum D {
    D0,
    D1(uint),
    D2 { x: (), y: () }
}

fn main() {
    macro_rules! t(
        ($ty:ty) => {{
            let x =rand::random::<$ty>();
            assert_eq!(x.to_str(), fmt!("%?", x));
        }}
    );

    for 20.times {
        t!(A);
        t!(B);
        t!(C);
        t!(D);
    }
}
