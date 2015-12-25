// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(panic_handler, const_fn, std_panic)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::panic;
use std::thread;

static A: AtomicUsize = AtomicUsize::new(0);
static B: AtomicUsize = AtomicUsize::new(0);

fn main() {
    panic::set_handler(|_| { A.fetch_add(1, Ordering::SeqCst); });
    let handler = panic::take_handler();
    panic::set_handler(move |info| {
        B.fetch_add(1, Ordering::SeqCst);
        handler(info);
    });

    let _ = thread::spawn(|| {
        panic!();
    }).join();

    assert_eq!(1, A.load(Ordering::SeqCst));
    assert_eq!(1, B.load(Ordering::SeqCst));
}
