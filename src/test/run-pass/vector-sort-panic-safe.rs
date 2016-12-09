// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

#![feature(rand)]
#![feature(const_fn)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::__rand::{thread_rng, Rng};
use std::thread;

const MAX_LEN: usize = 80;

static DROP_COUNTS: [AtomicUsize; MAX_LEN] = [
    // FIXME #5244: AtomicUsize is not Copy.
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
    AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0),
];

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
struct DropCounter {
    x: u32,
    id: usize,
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        DROP_COUNTS[self.id].fetch_add(1, Ordering::Relaxed);
    }
}

fn test(input: &[DropCounter]) {
    let len = input.len();

    // Work out the total number of comparisons required to sort
    // this array...
    let mut count = 0usize;
    input.to_owned().sort_by(|a, b| { count += 1; a.cmp(b) });

    // ... and then panic on each and every single one.
    for panic_countdown in 0..count {
        // Refresh the counters.
        for i in 0..len {
            DROP_COUNTS[i].store(0, Ordering::Relaxed);
        }

        let v = input.to_owned();
        let _ = thread::spawn(move || {
            let mut v = v;
            let mut panic_countdown = panic_countdown;
            v.sort_by(|a, b| {
                if panic_countdown == 0 {
                    panic!();
                }
                panic_countdown -= 1;
                a.cmp(b)
            })
        }).join();

        // Check that the number of things dropped is exactly
        // what we expect (i.e. the contents of `v`).
        for (i, c) in DROP_COUNTS.iter().enumerate().take(len) {
            let count = c.load(Ordering::Relaxed);
            assert!(count == 1,
                    "found drop count == {} for i == {}, len == {}",
                    count, i, len);
        }
    }
}

fn main() {
    for len in (1..20).chain(70..MAX_LEN) {
        // Test on a random array.
        let mut rng = thread_rng();
        let input = (0..len).map(|id| {
            DropCounter {
                x: rng.next_u32(),
                id: id,
            }
        }).collect::<Vec<_>>();
        test(&input);

        // Test on a sorted array with two elements randomly swapped, creating several natural
        // runs of random lengths. Such arrays have very high chances of hitting all code paths in
        // the merge procedure.
        for _ in 0..5 {
            let mut input = (0..len).map(|i|
                DropCounter {
                    x: i as u32,
                    id: i,
                }
            ).collect::<Vec<_>>();

            let a = rng.gen::<usize>() % len;
            let b = rng.gen::<usize>() % len;
            input.swap(a, b);

            test(&input);
        }
    }
}
