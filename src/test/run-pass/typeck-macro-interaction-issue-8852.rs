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

// after fixing #9384 and implementing hygiene for match bindings,
// this now fails because the insertion of the 'y' into the match
// doesn't cause capture. Making this macro hygienic (as I've done)
// could very well make this test case completely pointless....

macro_rules! test(
    ($id1:ident, $id2:ident, $e:expr) => (
        fn foo(a:T, b:T) -> T {
            match (a, b) {
                (A($id1), A($id2)) => A($e),
                (B($id1), B($id2)) => B($e),
                _ => fail!()
            }
        }
    )
)

test!(x,y,x + y)

pub fn main() {
    foo(A(1), A(2));
}
