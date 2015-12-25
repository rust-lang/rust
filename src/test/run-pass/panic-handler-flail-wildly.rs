// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![feature(panic_handler, std_panic)]

use std::panic;
use std::thread;

fn a() {
    panic::set_handler(|_| println!("hello yes this is a"));
    panic::take_handler();
    panic::set_handler(|_| println!("hello yes this is a part 2"));
    panic::take_handler();
}

fn b() {
    panic::take_handler();
    panic::take_handler();
    panic::take_handler();
    panic::take_handler();
    panic::take_handler();
    panic!();
}

fn c() {
    panic::set_handler(|_| ());
    panic::set_handler(|_| ());
    panic::set_handler(|_| ());
    panic::set_handler(|_| ());
    panic::set_handler(|_| ());
    panic::set_handler(|_| ());
    panic!();
}

fn main() {
    for _ in 0..10 {
        let mut handles = vec![];
        for _ in 0..10 {
            handles.push(thread::spawn(a));
        }
        for _ in 0..10 {
            handles.push(thread::spawn(b));
        }
        for _ in 0..10 {
            handles.push(thread::spawn(c));
        }
        for handle in handles {
            let _ = handle.join();
        }
    }
}
