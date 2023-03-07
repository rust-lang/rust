// run-pass
// needs-unwind
#![allow(stable_features)]

// ignore-emscripten no threads support

#![feature(std_panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::panic;
use std::thread;

static A: AtomicUsize = AtomicUsize::new(0);
static B: AtomicUsize = AtomicUsize::new(0);

fn main() {
    panic::set_hook(Box::new(|_| { A.fetch_add(1, Ordering::SeqCst); }));
    let hook = panic::take_hook();
    panic::set_hook(Box::new(move |info| {
        B.fetch_add(1, Ordering::SeqCst);
        hook(info);
    }));

    let _ = thread::spawn(|| {
        panic!();
    }).join();

    assert_eq!(1, A.load(Ordering::SeqCst));
    assert_eq!(1, B.load(Ordering::SeqCst));
}
