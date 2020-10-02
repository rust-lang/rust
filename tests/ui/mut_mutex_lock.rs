#![warn(clippy::mut_mutex_lock)]

use std::sync::{Arc, Mutex};

fn mut_mutex_lock() {
    let mut value_rc = Arc::new(Mutex::new(42_u8));
    let value_mutex = Arc::get_mut(&mut value_rc).unwrap();

    let value = value_mutex.lock().unwrap();
    *value += 1;
}

fn no_owned_mutex_lock() {
    let mut value_rc = Arc::new(Mutex::new(42_u8));
    let value = value_rc.lock().unwrap();
    *value += 1;
}

fn main() {}
