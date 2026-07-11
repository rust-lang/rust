// https://github.com/rust-lang/rust/issues/4734
//@ run-pass
#![allow(dead_code)]
// Ensures that destructors are run for expressions of the form "e;" where
// `e` is a type which requires a destructor.

#![allow(path_statements)]

use std::sync::atomic::{AtomicUsize, Ordering};

struct A { n: isize }
struct B;

static NUM_DROPS: AtomicUsize = AtomicUsize::new(0);

impl Drop for A {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for B {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 0);
    { let _a = A { n: 1 }; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 1);
    { A { n: 3 }; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 2);

    { let _b = B; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 3);
    { B; }
    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 4);
}
