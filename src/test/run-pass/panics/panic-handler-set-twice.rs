// run-pass
#![allow(unused_variables)]
#![allow(stable_features)]

#![feature(std_panic)]

// ignore-emscripten no threads support

use std::sync::atomic::{AtomicUsize, Ordering};
use std::panic;
use std::thread;

static A: AtomicUsize = AtomicUsize::new(0);

fn main() {
    panic::set_hook(Box::new(|_| ()));
    panic::set_hook(Box::new(|info| { A.fetch_add(1, Ordering::SeqCst); }));

    let _ = thread::spawn(|| {
        panic!();
    }).join();

    assert_eq!(1, A.load(Ordering::SeqCst));
}
