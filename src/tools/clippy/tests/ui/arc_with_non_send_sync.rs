//@aux-build:proc_macros.rs
#![warn(clippy::arc_with_non_send_sync)]
#![allow(unused_variables)]

#[macro_use]
extern crate proc_macros;

use std::cell::RefCell;
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

fn foo<T>(x: T) {
    // Should not lint - purposefully ignoring generic args.
    let a = Arc::new(x);
}
fn issue11076<T>() {
    let a: Arc<Vec<T>> = Arc::new(Vec::new());
}

fn issue11232() {
    external! {
        let a: Arc<*const u8> = Arc::new(null());
        let a: Arc<*mut u8> = Arc::new(null_mut());
    }
    with_span! {
        span
        let a: Arc<*const u8> = Arc::new(null());
        let a: Arc<*mut u8> = Arc::new(null_mut());
    }
}

fn main() {
    let _ = Arc::new(42);

    let _ = Arc::new(RefCell::new(42));
    //~^ ERROR: usage of an `Arc` that is not `Send` and `Sync`
    //~| NOTE: the trait `Sync` is not implemented for `RefCell<i32>`

    let mutex = Mutex::new(1);
    let _ = Arc::new(mutex.lock().unwrap());
    //~^ ERROR: usage of an `Arc` that is not `Send` and `Sync`
    //~| NOTE: the trait `Send` is not implemented for `MutexGuard<'_, i32>`

    let _ = Arc::new(&42 as *const i32);
    //~^ ERROR: usage of an `Arc` that is not `Send` and `Sync`
    //~| NOTE: the trait `Send` is not implemented for `*const i32`
    //~| NOTE: the trait `Sync` is not implemented for `*const i32`
}
