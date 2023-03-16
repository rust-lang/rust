//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@compile-flags: -Zmiri-strict-provenance

//! The main purpose of this test is to check that if we take a pointer to
//! thread's `t1` thread-local `A` and send it to another thread `t2`,
//! dereferencing the pointer on `t2` resolves to `t1`'s thread-local. In this
//! test, we also check that thread-locals act as per-thread statics.

#![feature(thread_local)]

use std::thread;

#[thread_local]
static mut A: u8 = 0;
#[thread_local]
static mut B: u8 = 0;
static mut C: u8 = 0;

// Regression test for https://github.com/rust-lang/rust/issues/96191.
#[thread_local]
static READ_ONLY: u8 = 42;

unsafe fn get_a_ref() -> *mut u8 {
    &mut A
}

struct Sender(*mut u8);

unsafe impl Send for Sender {}

fn main() {
    let _val = READ_ONLY;

    let ptr = unsafe {
        let x = get_a_ref();
        *x = 5;
        assert_eq!(A, 5);
        B = 15;
        C = 25;
        Sender(&mut A)
    };

    thread::spawn(move || unsafe {
        assert_eq!(*ptr.0, 5);
        assert_eq!(A, 0);
        assert_eq!(B, 0);
        assert_eq!(C, 25);
        B = 14;
        C = 24;
        let y = get_a_ref();
        assert_eq!(*y, 0);
        *y = 4;
        assert_eq!(*ptr.0, 5);
        assert_eq!(A, 4);
        assert_eq!(*get_a_ref(), 4);
    })
    .join()
    .unwrap();

    unsafe {
        assert_eq!(*get_a_ref(), 5);
        assert_eq!(A, 5);
        assert_eq!(B, 15);
        assert_eq!(C, 24);
    }
}
