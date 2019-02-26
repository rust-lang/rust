// run-pass

#![feature(generators, generator_trait)]

use std::ops::Generator;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};

static A: AtomicUsize = AtomicUsize::new(0);

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
    Pin::new(&mut a).resume();
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
    Pin::new(&mut a).resume();
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
    Pin::new(&mut a).resume();
    assert_eq!(A.load(Ordering::SeqCst), n);
    Pin::new(&mut a).resume();
    assert_eq!(A.load(Ordering::SeqCst), n + 1);
}
