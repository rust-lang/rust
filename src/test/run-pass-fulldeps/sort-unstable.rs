// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_private, sort_internals)]

extern crate core;
extern crate rand;

use std::cmp::Ordering::{Equal, Greater, Less};
use core::slice::heapsort;

use rand::{Rng, XorShiftRng};

fn main() {
    let mut v = [0; 600];
    let mut tmp = [0; 600];
    let mut rng = XorShiftRng::new_unseeded();

    for len in (2..25).chain(500..510) {
        let v = &mut v[0..len];
        let tmp = &mut tmp[0..len];

        for &modulus in &[5, 10, 100, 1000] {
            for _ in 0..100 {
                for i in 0..len {
                    v[i] = rng.gen::<i32>() % modulus;
                }

                // Sort in default order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable();
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Sort in ascending order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable_by(|a, b| a.cmp(b));
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Sort in descending order.
                tmp.copy_from_slice(v);
                tmp.sort_unstable_by(|a, b| b.cmp(a));
                assert!(tmp.windows(2).all(|w| w[0] >= w[1]));

                // Test heapsort using `<` operator.
                tmp.copy_from_slice(v);
                heapsort(tmp, |a, b| a < b);
                assert!(tmp.windows(2).all(|w| w[0] <= w[1]));

                // Test heapsort using `>` operator.
                tmp.copy_from_slice(v);
                heapsort(tmp, |a, b| a > b);
                assert!(tmp.windows(2).all(|w| w[0] >= w[1]));
            }
        }
    }

    // Sort using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    for i in 0..v.len() {
        v[i] = i as i32;
    }
    v.sort_unstable_by(|_, _| *rng.choose(&[Less, Equal, Greater]).unwrap());
    v.sort_unstable();
    for i in 0..v.len() {
        assert_eq!(v[i], i as i32);
    }

    // Should not panic.
    [0i32; 0].sort_unstable();
    [(); 10].sort_unstable();
    [(); 100].sort_unstable();

    let mut v = [0xDEADBEEFu64];
    v.sort_unstable();
    assert!(v == [0xDEADBEEF]);
}
