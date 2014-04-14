// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

enum T {
    A(int),
    B(f64)
}

macro_rules! test(
    ($e:expr) => (
        fn foo(a:T, b:T) -> T {
            match (a, b) {
                (A(x), A(y)) => A($e),
                (B(x), B(y)) => B($e),
                _ => fail!()
            }
        }
    )
)

test!(x + y)

pub fn main() {
    foo(A(1), A(2));
}
