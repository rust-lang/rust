//@ compile-flags: -Zunstable-options

//@revisions: edition2021 edition2024
//@[edition2021] edition:2021
//@[edition2024] edition:2024
//@[edition2024] check-pass
#![warn(clippy::if_let_mutex)]
#![allow(clippy::redundant_pattern_matching)]

use std::ops::Deref;
use std::sync::Mutex;

fn do_stuff<T>(_: T) {}

fn if_let() {
    let m = Mutex::new(1_u8);
    if let Err(locked) = m.lock() {
        //~[edition2021]^ if_let_mutex
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
        //~[edition2021]^ if_let_mutex
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
        //~[edition2021]^ if_let_mutex
        do_stuff(i);
    } else {
        let _x = mutex.lock();
    };
}

fn multiple_mutexes(m1: &Mutex<()>, m2: &Mutex<()>) {
    if let Ok(_) = m1.lock() {
        m2.lock();
    } else {
        m1.lock();
    }
    //~[edition2021]^^^^^ if_let_mutex
}

fn main() {}
