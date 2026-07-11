//@ run-pass
#![allow(unused_must_use)]
// Test that we are able to reinitialize box with moved referent

use std::sync::atomic::{AtomicUsize, Ordering};

static ORDER: [AtomicUsize; 3] = [const { AtomicUsize::new(0) }; 3];
static INDEX: AtomicUsize = AtomicUsize::new(0);

fn push_order(value: usize) {
    let index = INDEX.fetch_add(1, Ordering::Relaxed);
    ORDER[index].store(value, Ordering::Relaxed);
}

fn order() -> [usize; 3] {
    [
        ORDER[0].load(Ordering::Relaxed),
        ORDER[1].load(Ordering::Relaxed),
        ORDER[2].load(Ordering::Relaxed),
    ]
}

struct Dropee (usize);

impl Drop for Dropee {
    fn drop(&mut self) {
        push_order(self.0);
    }
}

fn add_sentintel() {
    push_order(2);
}

fn main() {
    let mut x = Box::new(Dropee(1));
    *x;  // move out from `*x`
    add_sentintel();
    *x = Dropee(3); // re-initialize `*x`
    {x}; // drop value
    assert_eq!(order(), [1, 2, 3]);
}
