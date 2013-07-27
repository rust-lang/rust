// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-pretty - token trees can't pretty print

// use test because #[should_fail] makes testing easier
// compile-flags:--test

// these are used for testing that `ignore` and `test_order` work.
#[deriving(Ord, TotalOrd, TotalEq)]
pub struct FailEq(uint);
impl Eq for FailEq {
    fn eq(&self, _: &FailEq) -> bool { fail!("eq") }
}

#[deriving(Eq, TotalOrd, TotalEq)]
pub struct FailOrd(uint);
impl Ord for FailOrd {
    fn lt(&self, _: &FailOrd) -> bool { fail!("lt") }
}

#[deriving(Eq, Ord, TotalOrd)]
pub struct FailTotalEq(uint);
impl TotalEq for FailTotalEq {
    fn equals(&self, _: &FailTotalEq) -> bool { fail!("equals") }
}

#[deriving(Eq, Ord, TotalEq)]
pub struct FailTotalOrd(uint);
impl TotalOrd for FailTotalOrd {
    fn cmp(&self, _: &FailTotalOrd) -> Ordering { fail!("cmp") }
}

// ignore the field that would fail if the given trait was called on
// it. (Only Eq and Ord get)
#[deriving(Eq(ignore(eq)),
           Ord(ignore(ord)))]
pub struct Ignore {
    eq: FailEq,
    ord: FailOrd,
}

#[test]
fn ignore() {
    let ignore = Ignore {
        eq: FailEq(1),
        ord: FailOrd(2),
    };


    ignore == ignore;
    ignore < ignore;
    ignore > ignore;
    ignore <= ignore;
    ignore >= ignore;
}

// Test `ok` first,
#[deriving(Eq(test_order(ok)),
           Ord(test_order(ok)),
           TotalEq(test_order(ok)),
           TotalOrd(test_order(ok)))]
pub struct SecondFirst<A> {
    failing: A,
    ok: uint
}

macro_rules! t_o_test {
    ($ctor:ident, $pass:expr, $fail:expr) => {
        mod $ctor {
            use super::*;

            #[test]
            fn test_pass() {
                // testing on `ok` will always all the implementation
                // to shortcircuit.
                let small = SecondFirst {
                    failing: $ctor(0),
                    ok: 0
                };
                let large = SecondFirst {
                    failing: $ctor(1),
                    ok: 1
                };

                $pass;
            }

            #[test]
            #[should_fail]
            fn test_fail() {
                // the implementations will have to test on `failing`,
                // (hopefully) making this fail.
                let small = SecondFirst {
                    failing: $ctor(0),
                    ok: 0
                };
                let large = small;
                $fail;
            }
        }
    }
}

t_o_test! { FailEq, small == large, small == large }
t_o_test! {
    FailOrd,
    {
        small < large;
        small > large;
        small <= large;
        small >= large;
    },
    {
        small < large;
        small > large;
        small <= large;
        small >= large;
    }
}
t_o_test! { FailTotalEq, small.equals(&large), small.equals(&large) }
t_o_test! { FailTotalOrd, small.cmp(&large), small.cmp(&large) }

#[deriving(Ord(reverse(x)),
           TotalEq, // satisfy TotalOrd's inheritance
           TotalOrd(reverse(x)))]
pub struct Reverse {
    x: uint
}

#[test]
fn reverse() {
    let small = Reverse { x: 0 };
    let large = Reverse { x: 1 };

    small > large;
    large < small;

    small <= small;
    small >= small;
    large <= large;
    large >= large;

    small >= large;
    large <= small;
}
