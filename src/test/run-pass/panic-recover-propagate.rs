// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

#![feature(panic_handler)]

use std::sync::atomic::{AtomicUsize, Ordering, ATOMIC_USIZE_INIT};
use std::panic;
use std::thread;

static A: AtomicUsize = ATOMIC_USIZE_INIT;

fn main() {
    panic::set_hook(Box::new(|_| {
        A.fetch_add(1, Ordering::SeqCst);
    }));

    let result = thread::spawn(|| {
        let result = panic::catch_unwind(|| {
            panic!("hi there");
        });

        panic::resume_unwind(result.unwrap_err());
    }).join();

    let msg = *result.unwrap_err().downcast::<&'static str>().unwrap();
    assert_eq!("hi there", msg);
    assert_eq!(1, A.load(Ordering::SeqCst));
}
