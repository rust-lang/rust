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

// This is the most common case as the above case is pretty
// contrived.
fn if_let_option() {
    let m = Mutex::new(Some(0_u8));
    if let Some(locked) = m.lock().unwrap().deref() {
        do_stuff(locked);
    } else {
        let lock = m.lock().unwrap();
        do_stuff(lock);
    };
}

// When mutexes are different don't warn
fn if_let_different_mutex() {
    let m = Mutex::new(Some(0_u8));
    let other = Mutex::new(None::<u8>);
    if let Some(locked) = m.lock().unwrap().deref() {
        do_stuff(locked);
    } else {
        let lock = other.lock().unwrap();
        do_stuff(lock);
    };
}

fn mutex_ref(mutex: &Mutex<i32>) {
    if let Ok(i) = mutex.lock() {
        do_stuff(i);
    } else {
        let _x = mutex.lock();
    };
}

fn main() {}
