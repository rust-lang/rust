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
use std::sync::atomic::{AtomicUint, INIT_ATOMIC_UINT, Relaxed};
use std::rand::{task_rng, Rng, Rand};

const REPEATS: uint = 5;
const MAX_LEN: uint = 32;
static drop_counts: [AtomicUint, .. MAX_LEN] =
    // FIXME #5244: AtomicUint is not Copy.
    [
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,

        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
        INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT, INIT_ATOMIC_UINT,
     ];

static creation_count: AtomicUint = INIT_ATOMIC_UINT;

#[deriving(Clone, PartialEq, PartialOrd, Eq, Ord)]
struct DropCounter { x: uint, creation_id: uint }

impl Rand for DropCounter {
    fn rand<R: Rng>(rng: &mut R) -> DropCounter {
        // (we're not using this concurrently, so Relaxed is fine.)
        let num = creation_count.fetch_add(1, Relaxed);
        DropCounter {
            x: rng.gen(),
            creation_id: num
        }
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        drop_counts[self.creation_id].fetch_add(1, Relaxed);
    }
}

pub fn main() {
    assert!(MAX_LEN <= std::uint::BITS);
    // len can't go above 64.
    for len in range(2, MAX_LEN) {
        for _ in range(0, REPEATS) {
            // reset the count for these new DropCounters, so their
            // IDs start from 0.
            creation_count.store(0, Relaxed);

            let main = task_rng().gen_iter::<DropCounter>()
                                 .take(len)
                                 .collect::<Vec<DropCounter>>();

            // work out the total number of comparisons required to sort
            // this array...
            let mut count = 0;
            main.clone().as_mut_slice().sort_by(|a, b| { count += 1; a.cmp(b) });

            // ... and then fail on each and every single one.
            for fail_countdown in range(0i, count) {
                // refresh the counters.
                for c in drop_counts.iter() {
                    c.store(0, Relaxed);
                }

                let v = main.clone();

                let _ = task::try(proc() {
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
                for (i, c) in drop_counts.iter().enumerate().take(len) {
                    let count = c.load(Relaxed);
                    assert!(count == 1,
                            "found drop count == {} for i == {}, len == {}",
                            count, i, len);
                }
            }
        }
    }
}
