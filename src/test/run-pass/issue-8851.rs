// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

// after fixing #9384 and implementing hygiene for match bindings,
// this now fails because the insertion of the 'y' into the match
// doesn't cause capture. Making this macro hygienic (as I've done)
// could very well make this test case completely pointless....

enum T {
    A(int),
    B(uint)
}

macro_rules! test(
    ($id:ident, $e:expr) => (
        fn foo(t: T) -> int {
            match t {
                A($id) => $e,
                B($id) => $e
            }
        }
    )
)

test!(y, 10 + (y as int))

pub fn main() {
    foo(A(20));
}
