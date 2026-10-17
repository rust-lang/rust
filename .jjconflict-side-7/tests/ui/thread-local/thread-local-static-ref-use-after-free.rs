//@ check-pass
//@ known-bug: #49682
//@ edition:2021

// Should fail. Keeping references to thread local statics can result in a
// use-after-free.

#![feature(thread_local)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

#[allow(dead_code)]
#[thread_local]
static FOO: AtomicUsize = AtomicUsize::new(0);

#[allow(dead_code)]
async fn bar() {}

#[allow(dead_code)]
async fn foo() {
    let r = &FOO;
    bar().await;
    r.load(Ordering::SeqCst);
}

fn main() {
    // &FOO = 0x7fd1e9cbf6d0
    _ = thread::spawn(|| {
        let g = foo();
        println!("&FOO = {:p}", &FOO);
        g
    })
    .join()
    .unwrap();

    // &FOO = 0x7fd1e9cc0f50
    println!("&FOO = {:p}", &FOO);

    // &FOO = 0x7fd1e9cbf6d0
    thread::spawn(move || {
        println!("&FOO = {:p}", &FOO);
    })
    .join()
    .unwrap();
}
