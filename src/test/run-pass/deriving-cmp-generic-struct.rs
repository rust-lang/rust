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
struct S<T> {
    x: T,
    y: T
}

pub fn main() {
    let s1 = S {x: 1, y: 1};
    let s2 = S {x: 1, y: 2};

    // in order for both Ord and TotalOrd
    let ss = [s1, s2];

    for (i, s1) in ss.iter().enumerate() {
        for (j, s2) in ss.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j;
            let le = i <= j;
            let gt = i > j;
            let ge = i >= j;

            // Eq
            fail_unless_eq!(*s1 == *s2, eq);
            fail_unless_eq!(*s1 != *s2, !eq);

            // TotalEq
            fail_unless_eq!(s1.equals(s2), eq);

            // Ord
            fail_unless_eq!(*s1 < *s2, lt);
            fail_unless_eq!(*s1 > *s2, gt);

            fail_unless_eq!(*s1 <= *s2, le);
            fail_unless_eq!(*s1 >= *s2, ge);

            // TotalOrd
            fail_unless_eq!(s1.cmp(s2), ord);
        }
    }
}
