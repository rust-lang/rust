#![warn(clippy::arc_with_non_send_sync)]
#![allow(unused_variables)]
use std::cell::RefCell;
use std::sync::{Arc, Mutex};

fn foo<T>(x: T) {
    // Should not lint - purposefully ignoring generic args.
    let a = Arc::new(x);
}
fn issue11076<T>() {
    let a: Arc<Vec<T>> = Arc::new(Vec::new());
}

fn main() {
    let _ = Arc::new(42);

    // !Sync
    let _ = Arc::new(RefCell::new(42));
    let mutex = Mutex::new(1);
    // !Send
    let _ = Arc::new(mutex.lock().unwrap());
    // !Send + !Sync
    let _ = Arc::new(&42 as *const i32);
}
