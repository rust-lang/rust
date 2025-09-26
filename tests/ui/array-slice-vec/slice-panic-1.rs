//@ run-pass
//@ needs-unwind
//@ needs-threads
//@ ignore-backends: gcc

// Test that if a slicing expr[..] fails, the correct cleanups happen.

// FIXME(static_mut_refs): this could use an atomic
#![allow(static_mut_refs)]

use std::thread;

struct Foo;

static mut DTOR_COUNT: isize = 0;

impl Drop for Foo {
    fn drop(&mut self) { unsafe { DTOR_COUNT += 1; } }
}

fn foo() {
    let x: &[_] = &[Foo, Foo];
    let _ = &x[3..4];
}

fn main() {
    let _ = thread::spawn(move|| foo()).join();
    unsafe { assert_eq!(DTOR_COUNT, 2); }
}
