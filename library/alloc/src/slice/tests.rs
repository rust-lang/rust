use crate::borrow::ToOwned;
use crate::rc::Rc;
use crate::string::ToString;
use crate::test_helpers::test_rng;
use crate::vec::Vec;

use core::cell::Cell;
use core::cmp::Ordering::{self, Equal, Greater, Less};
use core::convert::identity;
use core::fmt;
use core::mem;
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use rand::{distributions::Standard, prelude::*, Rng, RngCore};
use std::panic;

macro_rules! do_test {
    ($input:ident, $func:ident) => {
        let len = $input.len();

        // Work out the total number of comparisons required to sort
        // this array...
        let mut count = 0usize;
        $input.to_owned().$func(|a, b| {
            count += 1;
            a.cmp(b)
        });

        // ... and then panic on each and every single one.
        for panic_countdown in 0..count {
            // Refresh the counters.
            VERSIONS.store(0, Relaxed);
            for i in 0..len {
                DROP_COUNTS[i].store(0, Relaxed);
            }

            let v = $input.to_owned();
            let _ = std::panic::catch_unwind(move || {
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
            });

            // Check that the number of things dropped is exactly
            // what we expect (i.e., the contents of `v`).
            for (i, c) in DROP_COUNTS.iter().enumerate().take(len) {
                let count = c.load(Relaxed);
                assert!(count == 1, "found drop count == {} for i == {}, len == {}", count, i, len);
            }

            // Check that the most recent versions of values were dropped.
            assert_eq!(VERSIONS.load(Relaxed), 0);
        }
    };
}

const MAX_LEN: usize = 80;

static DROP_COUNTS: [AtomicUsize; MAX_LEN] = [
    // FIXME(RFC 1109): AtomicUsize is not Copy.
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
    AtomicUsize::new(0),
];

static VERSIONS: AtomicUsize = AtomicUsize::new(0);

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

std::thread_local!(static SILENCE_PANIC: Cell<bool> = Cell::new(false));

#[test]
#[cfg_attr(target_os = "emscripten", ignore)] // no threads
#[cfg_attr(not(panic = "unwind"), ignore = "test requires unwinding support")]
fn panic_safe() {
    panic::update_hook(move |prev, info| {
        if !SILENCE_PANIC.with(|s| s.get()) {
            prev(info);
        }
    });

    let mut rng = test_rng();

    // Miri is too slow (but still need to `chain` to make the types match)
    let lens = if cfg!(miri) { (1..10).chain(0..0) } else { (1..20).chain(70..MAX_LEN) };
    let moduli: &[u32] = if cfg!(miri) { &[5] } else { &[5, 20, 50] };

    for len in lens {
        for &modulus in moduli {
            for &has_runs in &[false, true] {
                let mut input = (0..len)
                    .map(|id| DropCounter {
                        x: rng.next_u32() % modulus,
                        id: id,
                        version: Cell::new(0),
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

                do_test!(input, sort_by);
                do_test!(input, sort_unstable_by);
            }
        }
    }

    // Set default panic hook again.
    drop(panic::take_hook());
}

#[test]
#[cfg_attr(miri, ignore)] // Miri is too slow
fn test_sort() {
    let mut rng = test_rng();

    for len in (2..25).chain(500..510) {
        for &modulus in &[5, 10, 100, 1000] {
            for _ in 0..10 {
                let orig: Vec<_> = (&mut rng)
                    .sample_iter::<i32, _>(&Standard)
                    .map(|x| x % modulus)
                    .take(len)
                    .collect();

                // Sort in default order.
                let mut v = orig.clone();
                v.sort();
                assert!(v.windows(2).all(|w| w[0] <= w[1]));

                // Sort in ascending order.
                let mut v = orig.clone();
                v.sort_by(|a, b| a.cmp(b));
                assert!(v.windows(2).all(|w| w[0] <= w[1]));

                // Sort in descending order.
                let mut v = orig.clone();
                v.sort_by(|a, b| b.cmp(a));
                assert!(v.windows(2).all(|w| w[0] >= w[1]));

                // Sort in lexicographic order.
                let mut v1 = orig.clone();
                let mut v2 = orig.clone();
                v1.sort_by_key(|x| x.to_string());
                v2.sort_by_cached_key(|x| x.to_string());
                assert!(v1.windows(2).all(|w| w[0].to_string() <= w[1].to_string()));
                assert!(v1 == v2);

                // Sort with many pre-sorted runs.
                let mut v = orig.clone();
                v.sort();
                v.reverse();
                for _ in 0..5 {
                    let a = rng.gen::<usize>() % len;
                    let b = rng.gen::<usize>() % len;
                    if a < b {
                        v[a..b].reverse();
                    } else {
                        v.swap(a, b);
                    }
                }
                v.sort();
                assert!(v.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }

    // Sort using a completely random comparison function.
    // This will reorder the elements *somehow*, but won't panic.
    let mut v = [0; 500];
    for i in 0..v.len() {
        v[i] = i as i32;
    }
    v.sort_by(|_, _| *[Less, Equal, Greater].choose(&mut rng).unwrap());
    v.sort();
    for i in 0..v.len() {
        assert_eq!(v[i], i as i32);
    }

    // Should not panic.
    [0i32; 0].sort();
    [(); 10].sort();
    [(); 100].sort();

    let mut v = [0xDEADBEEFu64];
    v.sort();
    assert!(v == [0xDEADBEEF]);
}

#[test]
fn test_sort_stability() {
    // Miri is too slow
    let large_range = if cfg!(miri) { 0..0 } else { 500..510 };
    let rounds = if cfg!(miri) { 1 } else { 10 };

    let mut rng = test_rng();
    for len in (2..25).chain(large_range) {
        for _ in 0..rounds {
            let mut counts = [0; 10];

            // create a vector like [(6, 1), (5, 1), (6, 2), ...],
            // where the first item of each tuple is random, but
            // the second item represents which occurrence of that
            // number this element is, i.e., the second elements
            // will occur in sorted order.
            let orig: Vec<_> = (0..len)
                .map(|_| {
                    let n = rng.gen::<usize>() % 10;
                    counts[n] += 1;
                    (n, counts[n])
                })
                .collect();

            let mut v = orig.clone();
            // Only sort on the first element, so an unstable sort
            // may mix up the counts.
            v.sort_by(|&(a, _), &(b, _)| a.cmp(&b));

            // This comparison includes the count (the second item
            // of the tuple), so elements with equal first items
            // will need to be ordered with increasing
            // counts... i.e., exactly asserting that this sort is
            // stable.
            assert!(v.windows(2).all(|w| w[0] <= w[1]));

            let mut v = orig.clone();
            v.sort_by_cached_key(|&(x, _)| x);
            assert!(v.windows(2).all(|w| w[0] <= w[1]));
        }
    }
}
