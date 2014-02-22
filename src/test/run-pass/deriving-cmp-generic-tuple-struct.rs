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
struct TS<T>(T,T);


pub fn main() {
    let ts1 = TS(1, 1);
    let ts2 = TS(1, 2);

    // in order for both Ord and TotalOrd
    let tss = [ts1, ts2];

    for (i, ts1) in tss.iter().enumerate() {
        for (j, ts2) in tss.iter().enumerate() {
            let ord = i.cmp(&j);

            let eq = i == j;
            let lt = i < j;
            let le = i <= j;
            let gt = i > j;
            let ge = i >= j;

            // Eq
            fail_unless_eq!(*ts1 == *ts2, eq);
            fail_unless_eq!(*ts1 != *ts2, !eq);

            // TotalEq
            fail_unless_eq!(ts1.equals(ts2), eq);

            // Ord
            fail_unless_eq!(*ts1 < *ts2, lt);
            fail_unless_eq!(*ts1 > *ts2, gt);

            fail_unless_eq!(*ts1 <= *ts2, le);
            fail_unless_eq!(*ts1 >= *ts2, ge);

            // TotalOrd
            fail_unless_eq!(ts1.cmp(ts2), ord);
        }
    }
}
