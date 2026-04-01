//! Panicking in other threads.

use std::thread;

fn panic() {
    let result = thread::spawn(|| panic!("Hello!")).join().unwrap_err();
    let msg = result.downcast_ref::<&'static str>().unwrap();
    assert_eq!(*msg, "Hello!");
}

fn panic_named() {
    thread::Builder::new()
        .name("childthread".to_string())
        .spawn(move || {
            panic!("Hello, world!");
        })
        .unwrap()
        .join()
        .unwrap_err();
}

fn main() {
    panic();
    panic_named();
}
