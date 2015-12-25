// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(std_panic, recover, panic_propagate, panic_handler, const_fn)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::panic;
use std::thread;

static A: AtomicUsize = AtomicUsize::new(0);

fn main() {
    panic::set_handler(|_| {
        A.fetch_add(1, Ordering::SeqCst);
    });

    let result = thread::spawn(|| {
        let result = panic::recover(|| {
            panic!("hi there");
        });

        panic::propagate(result.err().unwrap());
    }).join();

    let msg = *result.err().unwrap().downcast::<&'static str>().unwrap();
    assert_eq!("hi there", msg);
    assert_eq!(1, A.load(Ordering::SeqCst));
}
