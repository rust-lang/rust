// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_assignment)]

use std::sync::mpsc::channel;

pub fn main() { test00(); }

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let (tx, rx) = channel();
    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();
    tx.send(4).unwrap();
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    tx.send(5).unwrap();
    tx.send(6).unwrap();
    tx.send(7).unwrap();
    tx.send(8).unwrap();
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    r = rx.recv().unwrap();
    sum += r;
    println!("{}", r);
    assert_eq!(sum, 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
