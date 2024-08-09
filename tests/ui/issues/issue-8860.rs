//@ run-pass
#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};

static DROP: AtomicUsize = AtomicUsize::new(0);
static DROP_S: AtomicUsize = AtomicUsize::new(0);
static DROP_T: AtomicUsize = AtomicUsize::new(0);

struct S;
impl Drop for S {
    fn drop(&mut self) {
        DROP_S.fetch_add(1, Ordering::Relaxed);
        DROP.fetch_add(1, Ordering::Relaxed);
    }
}
fn f(ref _s: S) {}

struct T { i: isize }
impl Drop for T {
    fn drop(&mut self) {
        DROP_T.fetch_add(1, Ordering::Relaxed);
        DROP.fetch_add(1, Ordering::Relaxed);
    }
}
fn g(ref _t: T) {}

fn do_test() {
    let s = S;
    f(s);
    assert_eq!(1, DROP.load(Ordering::Relaxed));
    assert_eq!(1, DROP_S.load(Ordering::Relaxed));
    let t = T { i: 1 };
    g(t);
    assert_eq!(1, DROP_T.load(Ordering::Relaxed));
}

fn main() {
    do_test();
    assert_eq!(2, DROP.load(Ordering::Relaxed));
    assert_eq!(1, DROP_S.load(Ordering::Relaxed));
    assert_eq!(1, DROP_T.load(Ordering::Relaxed));
}
