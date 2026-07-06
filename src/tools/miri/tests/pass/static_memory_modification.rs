// FIXME(static_mut_refs): use raw pointers instead of references
#![allow(static_mut_refs)]

use std::sync::atomic::{AtomicUsize, Ordering};

static mut X: usize = 5;
static Y: AtomicUsize = AtomicUsize::new(5);

fn main() {
    unsafe {
        X = 6;
        assert_eq!(X, 6);
    }

    Y.store(6, Ordering::Relaxed);
    assert_eq!(Y.load(Ordering::Relaxed), 6);
}
