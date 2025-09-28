//@ run-pass
//@ needs-unwind
#![allow(stable_features)]

//@ needs-threads
//@ ignore-backends: gcc

#![feature(std_panic)]
#![feature(panic_update_hook)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::panic;
use std::thread;

static A: AtomicUsize = AtomicUsize::new(0);
static B: AtomicUsize = AtomicUsize::new(0);
static C: AtomicUsize = AtomicUsize::new(0);

fn main() {
    panic::set_hook(Box::new(|_| { A.fetch_add(1, Ordering::SeqCst); }));
    panic::update_hook(|prev, info| {
        B.fetch_add(1, Ordering::SeqCst);
        prev(info);
    });
    panic::update_hook(|prev, info| {
        C.fetch_add(1, Ordering::SeqCst);
        prev(info);
    });

    let _ = thread::spawn(|| {
        panic!();
    }).join();

    assert_eq!(1, A.load(Ordering::SeqCst));
    assert_eq!(1, B.load(Ordering::SeqCst));
    assert_eq!(1, C.load(Ordering::SeqCst));
}
