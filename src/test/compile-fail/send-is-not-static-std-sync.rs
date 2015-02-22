// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// basic tests to see that certain "obvious" errors are caught by
// these types no longer requiring `'static` (RFC 458)

#![allow(dead_code)]

use std::sync::{Mutex, RwLock, mpsc};

fn mutex() {
    let x = 1;
    let y = Box::new(1);
    let lock = Mutex::new(&x);
    *lock.lock().unwrap() = &*y;
    drop(y); //~ ERROR cannot move out
    {
        let z = 2;
        *lock.lock().unwrap() = &z; //~ ERROR does not live long enough
    }
}

fn rwlock() {
    let x = 1;
    let y = Box::new(1);
    let lock = RwLock::new(&x);
    *lock.write().unwrap() = &*y;
    drop(y); //~ ERROR cannot move out
    {
        let z = 2;
        *lock.write().unwrap() = &z; //~ ERROR does not live long enough
    }
}

fn channel() {
    let x = 1;
    let y = Box::new(1);
    let (tx, rx) = mpsc::channel();

    tx.send(&x).unwrap();
    tx.send(&*y);
    drop(y); //~ ERROR cannot move out
    {
        let z = 2;
        tx.send(&z).unwrap(); //~ ERROR does not live long enough
    }
}

fn main() {}
