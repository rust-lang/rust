// run-pass
#![allow(dead_code)]
// aux-build:issue-17718-aux.rs

extern crate issue_17718_aux as other;

use std::sync::atomic::{AtomicUsize, Ordering};

const C1: usize = 1;
const C2: AtomicUsize = AtomicUsize::new(0);
const C3: fn() = foo;
const C4: usize = C1 * C1 + C1 / C1;
const C5: &'static usize = &C4;
const C6: usize = {
    const C: usize = 3;
    C
};

static S1: usize = 3;
static S2: AtomicUsize = AtomicUsize::new(0);

mod test {
    static A: usize = 4;
    static B: &'static usize = &A;
    static C: &'static usize = &(A);
}

fn foo() {}

fn main() {
    assert_eq!(C1, 1);
    assert_eq!(C3(), ());
    assert_eq!(C2.fetch_add(1, Ordering::SeqCst), 0);
    assert_eq!(C2.fetch_add(1, Ordering::SeqCst), 0);
    assert_eq!(C4, 2);
    assert_eq!(*C5, 2);
    assert_eq!(C6, 3);
    assert_eq!(S1, 3);
    assert_eq!(S2.fetch_add(1, Ordering::SeqCst), 0);
    assert_eq!(S2.fetch_add(1, Ordering::SeqCst), 1);

    match 1 {
        C1 => {}
        _ => unreachable!(),
    }

    let _a = C1;
    let _a = C2;
    let _a = C3;
    let _a = C4;
    let _a = C5;
    let _a = C6;
    let _a = S1;

    assert_eq!(other::C1, 1);
    assert_eq!(other::C3(), ());
    assert_eq!(other::C2.fetch_add(1, Ordering::SeqCst), 0);
    assert_eq!(other::C2.fetch_add(1, Ordering::SeqCst), 0);
    assert_eq!(other::C4, 2);
    assert_eq!(*other::C5, 2);
    assert_eq!(other::S1, 3);
    assert_eq!(other::S2.fetch_add(1, Ordering::SeqCst), 0);
    assert_eq!(other::S2.fetch_add(1, Ordering::SeqCst), 1);

    let _a = other::C1;
    let _a = other::C2;
    let _a = other::C3;
    let _a = other::C4;
    let _a = other::C5;

    match 1 {
        other::C1 => {}
        _ => unreachable!(),
    }
}
