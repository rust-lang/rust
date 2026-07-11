//@ run-pass
#![allow(unused_assignments)]

// Drop works for union itself.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU8, Ordering};

struct S;

union U {
    a: ManuallyDrop<S>
}

impl Drop for S {
    fn drop(&mut self) {
        CHECK.fetch_add(10, Ordering::Relaxed);
    }
}

impl Drop for U {
    fn drop(&mut self) {
        CHECK.fetch_add(1, Ordering::Relaxed);
    }
}

static CHECK: AtomicU8 = AtomicU8::new(0);

fn main() {
    unsafe {
        let mut u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK.load(Ordering::Relaxed), 0);
        u = U { a: ManuallyDrop::new(S) };
        assert_eq!(CHECK.load(Ordering::Relaxed), 1); // union itself is assigned, union is dropped, field is not dropped
        *u.a = S;
        assert_eq!(CHECK.load(Ordering::Relaxed), 11); // union field is assigned, field is dropped
    }
}
