//! Regression test for https://github.com/rust-lang/rust/issues/162843.
//! Ignoring a synchronization guard should suggest a binding that keeps the lock held.
#![deny(unused_must_use)]

use std::sync::{Mutex, RwLock};

fn main() {
    let mutex = Mutex::new(0);
    mutex.lock().unwrap(); //~ ERROR unused `std::sync::MutexGuard` that must be used
    mutex.lock(); //~ ERROR unused `Result` that must be used

    let rwlock = RwLock::new(0);
    rwlock.read().unwrap(); //~ ERROR unused `std::sync::RwLockReadGuard` that must be used
    rwlock.write().unwrap(); //~ ERROR unused `std::sync::RwLockWriteGuard` that must be used
}
