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

#![feature(rustc_private)]
#![feature(sort_unstable)]

extern crate rand;

use rand::{thread_rng, Rng};
use std::cell::Cell;
use std::cmp::Ordering;
use std::panic;
use std::sync::atomic::{ATOMIC_USIZE_INIT, AtomicUsize};
use std::sync::atomic::Ordering::Relaxed;
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

static VERSIONS: AtomicUsize = ATOMIC_USIZE_INIT;

#[derive(Clone, Eq)]
struct DropCounter {
    x: u32,
    id: usize,
    version: Cell<usize>,
}

impl PartialEq for DropCounter {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd for DropCounter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.version.set(self.version.get() + 1);
        other.version.set(other.version.get() + 1);
        VERSIONS.fetch_add(2, Relaxed);
        self.x.partial_cmp(&other.x)
    }
}

impl Ord for DropCounter {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Drop for DropCounter {
    fn drop(&mut self) {
        DROP_COUNTS[self.id].fetch_add(1, Relaxed);
        VERSIONS.fetch_sub(self.version.get(), Relaxed);
    }
}

macro_rules! test {
    ($input:ident, $func:ident) => {
        let len = $input.len();

        // Work out the total number of comparisons required to sort
        // this array...
        let mut count = 0usize;
        $input.to_owned().$func(|a, b| { count += 1; a.cmp(b) });

        // ... and then panic on each and every single one.
        for panic_countdown in 0..count {
            // Refresh the counters.
            VERSIONS.store(0, Relaxed);
            for i in 0..len {
                DROP_COUNTS[i].store(0, Relaxed);
            }

            let v = $input.to_owned();
            let _ = thread::spawn(move || {
                let mut v = v;
                let mut panic_countdown = panic_countdown;
                v.$func(|a, b| {
                    if panic_countdown == 0 {
                        SILENCE_PANIC.with(|s| s.set(true));
                        panic!();
                    }
                    panic_countdown -= 1;
                    a.cmp(b)
                })
            }).join();

            // Check that the number of things dropped is exactly
            // what we expect (i.e. the contents of `v`).
            for (i, c) in DROP_COUNTS.iter().enumerate().take(len) {
                let count = c.load(Relaxed);
                assert!(count == 1,
                        "found drop count == {} for i == {}, len == {}",
                        count, i, len);
            }

            // Check that the most recent versions of values were dropped.
            assert_eq!(VERSIONS.load(Relaxed), 0);
        }
    }
}

thread_local!(static SILENCE_PANIC: Cell<bool> = Cell::new(false));

fn main() {
    let prev = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        if !SILENCE_PANIC.with(|s| s.get()) {
            prev(info);
        }
    }));

    let mut rng = thread_rng();

    for len in (1..20).chain(70..MAX_LEN) {
        for &modulus in &[5, 20, 50] {
            for &has_runs in &[false, true] {
                let mut input = (0..len)
                    .map(|id| {
                        DropCounter {
                            x: rng.next_u32() % modulus,
                            id: id,
                            version: Cell::new(0),
                        }
                    })
                    .collect::<Vec<_>>();

                if has_runs {
                    for c in &mut input {
                        c.x = c.id as u32;
                    }

                    for _ in 0..5 {
                        let a = rng.gen::<usize>() % len;
                        let b = rng.gen::<usize>() % len;
                        if a < b {
                            input[a..b].reverse();
                        } else {
                            input.swap(a, b);
                        }
                    }
                }

                test!(input, sort_by);
                test!(input, sort_unstable_by);
            }
        }
    }
}
