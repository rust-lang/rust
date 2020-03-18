#![warn(clippy::if_let_mutex)]

use std::sync::Mutex;

fn do_stuff<T>(_: T) {}
fn foo() {
    let m = Mutex::new(1u8);

    if let Err(locked) = m.lock() {
        do_stuff(locked);
    } else {
        let lock = m.lock().unwrap();
        do_stuff(lock);
    };
}

fn main() {}
