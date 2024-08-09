//@ run-pass
//@ needs-unwind

//@ ignore-emscripten no threads support

// Test that if a slicing expr[..] fails, the correct cleanups happen.


use std::thread;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Foo;

static DTOR_COUNT: AtomicUsize = AtomicUsize::new(0);

impl Drop for Foo {
    fn drop(&mut self) {
        DTOR_COUNT.fetch_add(1, Ordering::Relaxed);
    }
}

fn bar() -> usize {
    panic!();
}

fn foo() {
    let x: &[_] = &[Foo, Foo];
    let _ = &x[3..bar()];
}

fn main() {
    let _ = thread::spawn(move|| foo()).join();
    assert_eq!(DTOR_COUNT.load(Ordering::Relaxed), 2);
}
