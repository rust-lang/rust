// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

#![feature(std_misc)]
#![allow(dead_assignment)]

use std::sync::mpsc::{channel, Sender};
use std::thread;

pub fn main() { test00(); }

fn test00_start(c: &Sender<isize>, start: isize,
                number_of_messages: isize) {
    let mut i: isize = 0;
    while i < number_of_messages { c.send(start + i).unwrap(); i += 1; }
}

fn test00() {
    let mut r: isize = 0;
    let mut sum: isize = 0;
    let (tx, rx) = channel();
    let number_of_messages: isize = 10;

    let tx2 = tx.clone();
    let t1 = thread::spawn(move|| {
        test00_start(&tx2, number_of_messages * 0, number_of_messages);
    });
    let tx2 = tx.clone();
    let t2 = thread::spawn(move|| {
        test00_start(&tx2, number_of_messages * 1, number_of_messages);
    });
    let tx2 = tx.clone();
    let t3 = thread::spawn(move|| {
        test00_start(&tx2, number_of_messages * 2, number_of_messages);
    });
    let tx2 = tx.clone();
    let t4 = thread::spawn(move|| {
        test00_start(&tx2, number_of_messages * 3, number_of_messages);
    });

    let mut i: isize = 0;
    while i < number_of_messages {
        r = rx.recv().unwrap();
        sum += r;
        r = rx.recv().unwrap();
        sum += r;
        r = rx.recv().unwrap();
        sum += r;
        r = rx.recv().unwrap();
        sum += r;
        i += 1;
    }

    assert_eq!(sum, number_of_messages * 4 * (number_of_messages * 4 - 1) / 2);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
}
