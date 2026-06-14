//@ run-pass

#![feature(partial_init_locals)]

use std::sync::atomic::{AtomicUsize, Ordering};

static COUNTER: AtomicUsize = AtomicUsize::new(0);

struct Droppy;

impl Drop for Droppy {
    fn drop(&mut self) {
        COUNTER.fetch_add(1, Ordering::Relaxed);
    }
}

fn conditional(cond: bool) {
    {
        assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
        struct A(Droppy);
        let a: A;
        if cond {
            a.0 = Droppy;
        }
    }
    assert_eq!(COUNTER.load(Ordering::Relaxed), if cond { 1 } else { 0 });
}

fn main() {
    let mut a = 0;
    conditional(a > 0);
    COUNTER.store(0, Ordering::Relaxed);
    a = 4;
    conditional(a > 0);
}
