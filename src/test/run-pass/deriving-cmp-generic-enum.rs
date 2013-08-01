// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Eq, TotalEq, Ord, TotalOrd)]
enum E<T> {
    E0,
    E1(T),
    E2(T,T)
}

pub fn main() {
    let e0 = E0;
    let e11 = E1(1);
    let e12 = E1(2);
    let e21 = E2(1, 1);
    let e22 = E2(1, 2);

    // in order for both Ord and TotalOrd
    let es = [e0, e11, e12, e21, e22];

    foreach (i, e1) in es.iter().enumerate() {
        foreach (j, e2) in es.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j;
            let le = i <= j;
            let gt = i > j;
            let ge = i >= j;

            // Eq
            assert_eq!(*e1 == *e2, eq);
            assert_eq!(*e1 != *e2, !eq);

            // TotalEq
            assert_eq!(e1.equals(e2), eq);

            // Ord
            assert_eq!(*e1 < *e2, lt);
            assert_eq!(*e1 > *e2, gt);

            assert_eq!(*e1 <= *e2, le);
            assert_eq!(*e1 >= *e2, ge);

            // TotalOrd
            assert_eq!(e1.cmp(e2), ord);
        }
    }
}
