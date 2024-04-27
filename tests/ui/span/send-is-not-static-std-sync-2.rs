// basic tests to see that certain "obvious" errors are caught by
// these types no longer requiring `'static` (RFC 458)

#![allow(dead_code)]

use std::sync::{Mutex, RwLock, mpsc};

fn mutex() {
    let lock = {
        let x = 1;
        Mutex::new(&x)
    };
    //~^^ ERROR `x` does not live long enough

    let _dangling = *lock.lock().unwrap();
}

fn rwlock() {
    let lock = {
        let x = 1;
        RwLock::new(&x)
    };
    //~^^ ERROR `x` does not live long enough
    let _dangling = *lock.read().unwrap();
}

fn channel() {
    let (_tx, rx) = {
        let x = 1;
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(&x);
        (tx, rx)
    };
    //~^^^ ERROR `x` does not live long enough

    let _dangling = rx.recv();
}

fn main() {}
