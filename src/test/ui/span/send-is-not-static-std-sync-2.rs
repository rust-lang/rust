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
    let lock = {
        let x = 1;
        Mutex::new(&x) //~ ERROR does not live long enough
    };

    let _dangling = *lock.lock().unwrap();
}

fn rwlock() {
    let lock = {
        let x = 1;
        RwLock::new(&x) //~ ERROR does not live long enough
    };
    let _dangling = *lock.read().unwrap();
}

fn channel() {
    let (_tx, rx) = {
        let x = 1;
        let (tx, rx) = mpsc::channel();
        let _ = tx.send(&x); //~ ERROR does not live long enough
        (tx, rx)
    };

    let _dangling = rx.recv();
}

fn main() {}
