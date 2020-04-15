#![warn(clippy::if_let_mutex)]

use std::ops::Deref;
use std::sync::Mutex;

fn do_stuff<T>(_: T) {}

fn if_let() {
    let m = Mutex::new(1_u8);
    if let Err(locked) = m.lock() {
        do_stuff(locked);
    } else {
        let lock = m.lock().unwrap();
        do_stuff(lock);
    };
}

fn if_let_option() {
    let m = Mutex::new(Some(0_u8));
    if let Some(locked) = m.lock().unwrap().deref() {
        do_stuff(locked);
    } else {
        let lock = m.lock().unwrap();
        do_stuff(lock);
    };
}

fn main() {}
