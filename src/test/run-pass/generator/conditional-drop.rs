// run-pass

#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::sync::atomic::{AtomicUsize, ATOMIC_USIZE_INIT, Ordering};

static A: AtomicUsize = ATOMIC_USIZE_INIT;

struct B;

impl Drop for B {
    fn drop(&mut self) {
        A.fetch_add(1, Ordering::SeqCst);
    }
}


fn test() -> bool { true }
fn test2() -> bool { false }

fn main() {
    t1();
    t2();
}

fn t1() {
    let mut a = || {
        let b = B;
        if test() {
            drop(b);
        }
        yield;
    };

    let n = A.load(Ordering::SeqCst);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
}

fn t2() {
    let mut a = || {
        let b = B;
        if test2() {
            drop(b);
        }
        yield;
    };

    let n = A.load(Ordering::SeqCst);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n);
    unsafe { a.resume() };
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
}
