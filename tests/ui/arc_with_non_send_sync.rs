#![warn(clippy::arc_with_non_send_sync)]
#![allow(unused_variables)]
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

fn foo<T>(x: T) {
    // Should not lint - purposefully ignoring generic args.
    let a = Arc::new(x);
}

fn main() {
    // This is safe, as `i32` implements `Send` and `Sync`.
    let a = Arc::new(42);

    // This is not safe, as `RefCell` does not implement `Sync`.
    let b = Arc::new(RefCell::new(42));
}
