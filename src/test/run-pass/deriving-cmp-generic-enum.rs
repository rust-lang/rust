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
    let e0 = E0, e11 = E1(1), e12 = E1(2), e21 = E2(1,1), e22 = E2(1, 2);

    // in order for both Ord and TotalOrd
    let es = [e0, e11, e12, e21, e22];

    for es.eachi |i, e1| {
        for es.eachi |j, e2| {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j, le = i <= j;
            let gt = i > j, ge = i >= j;

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
