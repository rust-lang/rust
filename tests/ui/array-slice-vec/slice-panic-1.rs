//@ run-pass
//@ needs-unwind
//@ needs-threads
//@ ignore-backends: gcc

// Test that if a slicing expr[..] fails, the correct cleanups happen.

use std::sync::atomic::{AtomicIsize, Ordering};
use std::thread;

struct Foo;

static DTOR_COUNT: AtomicIsize = AtomicIsize::new(0);

impl Drop for Foo {
    fn drop(&mut self) { DTOR_COUNT.fetch_add(1, Ordering::Relaxed); }
}

fn foo() {
    let x: &[_] = &[Foo, Foo];
    let _ = &x[3..4];
}

fn main() {
    let _ = thread::spawn(move|| foo()).join();
    assert_eq!(DTOR_COUNT.load(Ordering::Relaxed), 2);
}
