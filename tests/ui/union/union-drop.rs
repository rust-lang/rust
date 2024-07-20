//@ run-pass

#![allow(dead_code)]
#![allow(unused_variables)]

use std::sync::atomic::{AtomicUsize, Ordering};

// Drop works for union itself.

#[derive(Copy, Clone)]
struct S;

union U {
    a: u8
}

union W {
    a: S,
}

union Y {
    a: S,
}

impl Drop for U {
    fn drop(&mut self) {
        CHECK.fetch_add(1, Ordering::Relaxed);
    }
}

impl Drop for W {
    fn drop(&mut self) {
        CHECK.fetch_add(1, Ordering::Relaxed);
    }
}

static CHECK: AtomicUsize = AtomicUsize::new(0);

fn main() {
    assert_eq!(CHECK.load(Ordering::Relaxed), 0);
    {
        let u = U { a: 1 };
    }
    assert_eq!(CHECK.load(Ordering::Relaxed), 1); // 1, dtor of U is called
    {
        let w = W { a: S };
    }
    assert_eq!(CHECK.load(Ordering::Relaxed), 2); // 2, dtor of W is called
    {
        let y = Y { a: S };
    }
    assert_eq!(CHECK.load(Ordering::Relaxed), 2); // 2, Y has no dtor
    {
        let u2 = U { a: 1 };
        std::mem::forget(u2);
    }
    assert_eq!(CHECK.load(Ordering::Relaxed), 2); // 2, dtor of U *not* called for u2
}
