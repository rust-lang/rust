// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast
// aux-build:trait_default_method_xc_aux.rs

extern crate aux = "trait_default_method_xc_aux";
use aux::{A, TestEquality, Something};
use aux::B;

fn f<T: aux::A>(i: T) {
    fail_unless_eq!(i.g(), 10);
}

fn welp<T>(i: int, _x: &T) -> int {
    i.g()
}

mod stuff {
    pub struct thing { x: int }
}

impl A for stuff::thing {
    fn f(&self) -> int { 10 }
}

fn g<T, U, V: B<T>>(i: V, j: T, k: U) -> (T, U) {
    i.thing(j, k)
}

fn eq<T: TestEquality>(lhs: &T, rhs: &T) -> bool {
    lhs.test_eq(rhs)
}
fn neq<T: TestEquality>(lhs: &T, rhs: &T) -> bool {
    lhs.test_neq(rhs)
}


impl TestEquality for stuff::thing {
    fn test_eq(&self, rhs: &stuff::thing) -> bool {
        //self.x.test_eq(&rhs.x)
        eq(&self.x, &rhs.x)
    }
}


pub fn main () {
    // Some tests of random things
    f(0);

    fail_unless_eq!(A::lurr(&0, &1), 21);

    let a = stuff::thing { x: 0 };
    let b = stuff::thing { x: 1 };
    let c = Something { x: 1 };

    fail_unless_eq!(0i.g(), 10);
    fail_unless_eq!(a.g(), 10);
    fail_unless_eq!(a.h(), 11);
    fail_unless_eq!(c.h(), 11);

    fail_unless_eq!(0i.thing(3.14, 1), (3.14, 1));
    fail_unless_eq!(B::staticthing(&0i, 3.14, 1), (3.14, 1));
    fail_unless_eq!(B::<f64>::staticthing::<int>(&0i, 3.14, 1), (3.14, 1));

    fail_unless_eq!(g(0i, 3.14, 1), (3.14, 1));
    fail_unless_eq!(g(false, 3.14, 1), (3.14, 1));

    let obj = ~0i as ~A;
    fail_unless_eq!(obj.h(), 11);


    // Trying out a real one
    fail_unless!(12.test_neq(&10));
    fail_unless!(!10.test_neq(&10));
    fail_unless!(a.test_neq(&b));
    fail_unless!(!a.test_neq(&a));

    fail_unless!(neq(&12, &10));
    fail_unless!(!neq(&10, &10));
    fail_unless!(neq(&a, &b));
    fail_unless!(!neq(&a, &a));
}
