// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:trait_default_method_xc_aux.rs

extern crate aux = "trait_default_method_xc_aux";
use aux::{A, TestEquality, Something};
use aux::B;

fn f<T: aux::A>(i: T) {
    assert_eq!(i.g(), 10);
}

fn welp<T>(i: int, _x: &T) -> int {
    i.g()
}

mod stuff {
    pub struct thing { pub x: int }
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


pub fn main() {
    // Some tests of random things
    f(0i);

    assert_eq!(A::lurr(&0i, &1i), 21);

    let a = stuff::thing { x: 0 };
    let b = stuff::thing { x: 1 };
    let c = Something { x: 1 };

    assert_eq!(0i.g(), 10);
    assert_eq!(a.g(), 10);
    assert_eq!(a.h(), 11);
    assert_eq!(c.h(), 11);

    assert_eq!(0i.thing(3.14f64, 1i), (3.14f64, 1i));
    assert_eq!(B::staticthing(&0i, 3.14f64, 1i), (3.14f64, 1i));
    assert_eq!(B::<f64>::staticthing::<int>(&0i, 3.14, 1), (3.14, 1));

    assert_eq!(g(0i, 3.14f64, 1i), (3.14f64, 1i));
    assert_eq!(g(false, 3.14f64, 1i), (3.14, 1));

    let obj = box 0i as Box<A>;
    assert_eq!(obj.h(), 11);


    // Trying out a real one
    assert!(12i.test_neq(&10i));
    assert!(!10i.test_neq(&10i));
    assert!(a.test_neq(&b));
    assert!(!a.test_neq(&a));

    assert!(neq(&12i, &10i));
    assert!(!neq(&10i, &10i));
    assert!(neq(&a, &b));
    assert!(!neq(&a, &a));
}
