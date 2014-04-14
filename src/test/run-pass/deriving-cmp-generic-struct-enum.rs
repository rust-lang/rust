// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(struct_variant)]

#[deriving(Eq, TotalEq, Ord, TotalOrd)]
enum ES<T> {
    ES1 { x: T },
    ES2 { x: T, y: T }
}


pub fn main() {
    let (es11, es12, es21, es22) = (ES1 {x: 1}, ES1 {x: 2}, ES2 {x: 1, y: 1}, ES2 {x: 1, y: 2});

    // in order for both Ord and TotalOrd
    let ess = [es11, es12, es21, es22];

    for (i, es1) in ess.iter().enumerate() {
        for (j, es2) in ess.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let (lt, le) = (i < j, i <= j);
            let (gt, ge) = (i > j, i >= j);

            // Eq
            assert_eq!(*es1 == *es2, eq);
            assert_eq!(*es1 != *es2, !eq);

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
