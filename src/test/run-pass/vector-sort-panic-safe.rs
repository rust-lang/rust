// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(rand, core)]

use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};
use std::__rand::{thread_rng, Rng};
use std::thread;

const REPEATS: usize = 5;
const MAX_LEN: usize = 32;
static drop_counts: [AtomicUsize;  MAX_LEN] =
    // FIXME #5244: AtomicUsize is not Copy.
    [
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
        ATOMIC_USIZE_INIT, ATOMIC_USIZE_INIT,
     ];

static creation_count: AtomicUsize = ATOMIC_USIZE_INIT;

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
struct DropCounter { x: u32, creation_id: usize }

impl Drop for DropCounter {
    fn drop(&mut self) {
        drop_counts[self.creation_id].fetch_add(1, Ordering::Relaxed);
    }
}

pub fn main() {
    assert!(MAX_LEN <= std::usize::BITS);
    // len can't go above 64.
    for len in 2..MAX_LEN {
        for _ in 0..REPEATS {
            // reset the count for these new DropCounters, so their
            // IDs start from 0.
            creation_count.store(0, Ordering::Relaxed);

            let mut rng = thread_rng();
            let main = (0..len).map(|_| {
                DropCounter {
                    x: rng.next_u32(),
                    creation_id: creation_count.fetch_add(1, Ordering::Relaxed),
                }
            }).collect::<Vec<_>>();

            // work out the total number of comparisons required to sort
            // this array...
            let mut count = 0_usize;
            main.clone().sort_by(|a, b| { count += 1; a.cmp(b) });

            // ... and then panic on each and every single one.
            for panic_countdown in 0..count {
                // refresh the counters.
                for c in &drop_counts {
                    c.store(0, Ordering::Relaxed);
                }

                let v = main.clone();

                let _ = thread::spawn(move|| {
                    let mut v = v;
                    let mut panic_countdown = panic_countdown;
                    v.sort_by(|a, b| {
                        if panic_countdown == 0 {
                            panic!()
                        }
                        panic_countdown -= 1;
                        a.cmp(b)
                    })
                }).join();

                // check that the number of things dropped is exactly
                // what we expect (i.e. the contents of `v`).
                for (i, c) in drop_counts.iter().enumerate().take(len) {
                    let count = c.load(Ordering::Relaxed);
                    assert!(count == 1,
                            "found drop count == {} for i == {}, len == {}",
                            count, i, len);
                }
            }
        }
    }
}
