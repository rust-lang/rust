// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rand::{Rng, task_rng};


#[test]
fn test_quicksort() {
    for len in range(4u, 25) {
        for _ in range(0i, 100) {
            let mut v = task_rng().gen_iter::<uint>().take(len)
                                  .collect::<Vec<uint>>();
            let mut v1 = v.clone();

            v.as_mut_slice().quicksort();
            assert!(v.as_slice().windows(2).all(|w| w[0] <= w[1]));

            v1.as_mut_slice().quicksort_by(|a, b| a.cmp(b));
            assert!(v1.as_slice().windows(2).all(|w| w[0] <= w[1]));

            v1.as_mut_slice().quicksort_by(|a, b| b.cmp(a));
            assert!(v1.as_slice().windows(2).all(|w| w[0] >= w[1]));
        }
    }

    // shouldn't fail/crash
    let mut v: [uint, .. 0] = [];
    v.quicksort();

    let mut v = [0xDEADBEEFu];
    v.quicksort();
    assert!(v == [0xDEADBEEF]);
}
