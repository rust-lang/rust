// revisions: both_off just_prop both_on
// ignore-tidy-linelength
// run-pass
// [both_off]  compile-flags: -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [just_prop] compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture
// edition:2018

#![feature(atomic_from_mut)]

// FIXME: I should be able to reduce the below further now that I understand the
// nature of the problem. (Namely, the fact that the old upvar_to_local_prop
// code was ignoring locals in projections.)

use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};

fn main() {
    let mut some_ptrs = [null_mut::<String>(); 10];
    let a = &*AtomicPtr::from_mut_slice(&mut some_ptrs);
    std::thread::scope(|s| {
        for i in 0..a.len() {
            s.spawn(move || {
                let name = Box::new(format!("thread{i}"));
                a[i].store(Box::into_raw(name), Ordering::Relaxed);
            });
        }
    });
    for p in some_ptrs {
        assert!(!p.is_null());
        let name = unsafe { Box::from_raw(p) };
        println!("Hello, {name}!");
    }
}
