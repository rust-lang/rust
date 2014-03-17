// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

use std::task;

pub fn main() { test00(); }

fn test00_start(c: &Sender<int>, number_of_messages: int) {
    let mut i: int = 0;
    while i < number_of_messages { c.send(i + 0); i += 1; }
}

fn test00() {
    let r: int = 0;
    let mut sum: int = 0;
    let (tx, rx) = channel();
    let number_of_messages: int = 10;

    let mut builder = task::task();
    let result = builder.future_result();
    builder.spawn(proc() {
        test00_start(&tx, number_of_messages);
    });

    let mut i: int = 0;
    while i < number_of_messages {
        sum += rx.recv();
        println!("{:?}", r);
        i += 1;
    }

    result.recv();

    assert_eq!(sum, number_of_messages * (number_of_messages - 1) / 2);
}
