// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// Tests for if as expressions with dynamic type sizes
type compare<T> = |T, T|: 'static -> bool;

fn test_generic<T:Clone>(expected: T, not_expected: T, eq: compare<T>) {
    let actual: T = if true { expected.clone() } else { not_expected };
    assert!((eq(expected, actual)));
}

fn test_bool() {
    fn compare_bool(b1: bool, b2: bool) -> bool { return b1 == b2; }
    test_generic::<bool>(true, false, compare_bool);
}

#[deriving(Clone)]
struct Pair {
    a: int,
    b: int,
}

fn test_rec() {
    fn compare_rec(t1: Pair, t2: Pair) -> bool {
        t1.a == t2.a && t1.b == t2.b
    }
    test_generic::<Pair>(Pair{a: 1, b: 2}, Pair{a: 2, b: 3}, compare_rec);
}

pub fn main() { test_bool(); test_rec(); }
