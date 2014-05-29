// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;
use std::rand::{task_rng, Rng};

static MAX_LEN: uint = 20;
static mut drop_counts: [uint, .. MAX_LEN] = [0, .. MAX_LEN];
static mut clone_count: uint = 0;

#[deriving(Rand, Eq, Ord, TotalEq, TotalOrd)]
struct DropCounter { x: uint, clone_num: uint }

impl Clone for DropCounter {
    fn clone(&self) -> DropCounter {
        let num = unsafe { clone_count };
        unsafe { clone_count += 1; }
        DropCounter {
            x: self.x,
            clone_num: num
        }
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        unsafe {
            // Rand creates some with arbitrary clone_nums
            if self.clone_num < MAX_LEN {
                drop_counts[self.clone_num] += 1;
            }
        }
    }
}

pub fn main() {
    // len can't go above 64.
    for len in range(2u, MAX_LEN) {
        for _ in range(0, 10) {
            let main = task_rng().gen_iter::<DropCounter>()
                                 .take(len)
                                 .collect::<Vec<DropCounter>>();

            // work out the total number of comparisons required to sort
            // this array...
            let mut count = 0;
            main.clone().as_mut_slice().sort_by(|a, b| { count += 1; a.cmp(b) });

            // ... and then fail on each and every single one.
            for fail_countdown in range(0, count) {
                // refresh the counters.
                unsafe {
                    drop_counts = [0, .. MAX_LEN];
                    clone_count = 0;
                }

                let v = main.clone();

                task::try(proc() {
                        let mut v = v;
                        let mut fail_countdown = fail_countdown;
                        v.as_mut_slice().sort_by(|a, b| {
                                if fail_countdown == 0 {
                                    fail!()
                                }
                                fail_countdown -= 1;
                                a.cmp(b)
                            })
                    });

                // check that the number of things dropped is exactly
                // what we expect (i.e. the contents of `v`).
                unsafe {
                    for (i, &c) in drop_counts.iter().enumerate() {
                        let expected = if i < len {1} else {0};
                        assert!(c == expected,
                                "found drop count == {} for i == {}, len == {}",
                                c, i, len);
                    }
                }
            }
        }
    }
}
