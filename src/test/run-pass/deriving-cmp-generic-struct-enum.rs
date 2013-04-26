// xfail-test #5530

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
enum ES<T> {
    ES1 { x: T },
    ES2 { x: T, y: T }
}


pub fn main() {
    let es11 = ES1 {x: 1}, es12 = ES1 {x: 2}, es21 = ES2 {x: 1, y: 1}, es22 = ES2 {x: 1, y: 2};

    // in order for both Ord and TotalOrd
    let ess = [es11, es12, es21, es22];

    for ess.eachi |i, es1| {
        for ess.eachi |j, es2| {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j, le = i <= j;
            let gt = i > j, ge = i >= j;

            // Eq
            assert_eq!(*es1 == *es2, eq);
            assert_eq!(*es1 != *es2, !eq);

            // TotalEq
            assert_eq!(es1.equals(es2), eq);

            // Ord
            assert_eq!(*es1 < *es2, lt);
            assert_eq!(*es1 > *es2, gt);

            assert_eq!(*es1 <= *es2, le);
            assert_eq!(*es1 >= *es2, ge);

            // TotalOrd
            assert_eq!(es1.cmp(es2), ord);
        }
    }
}