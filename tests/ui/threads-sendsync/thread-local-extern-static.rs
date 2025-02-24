//@ run-pass
//@ ignore-windows FIXME(134939): thread_local + no_mangle doesn't work on Windows
//@ aux-build:thread-local-extern-static.rs

#![feature(cfg_target_thread_local, thread_local)]

#[cfg(target_thread_local)]
extern crate thread_local_extern_static;

#[cfg(target_thread_local)]
use std::cell::Cell;

#[cfg(target_thread_local)]
extern "C" {
    #[thread_local]
    static FOO: Cell<u32>;
}

#[cfg(target_thread_local)]
fn main() {
    unsafe {
        assert_eq!(FOO.get(), 3);
    }
}

#[cfg(not(target_thread_local))]
fn main() {}
