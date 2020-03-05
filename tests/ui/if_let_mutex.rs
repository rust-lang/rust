#![warn(clippy::if_let_mutex)]

use std::sync::Mutex;

fn do_stuff() {}
fn foo() {
    let m = Mutex::new(1u8);

    if let Ok(locked) = m.lock() {
        do_stuff();
    } else {
        m.lock().unwrap();
    };
}

fn main() {}
