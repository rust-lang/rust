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

pub fn main() { test00(); }

fn test00() {
    let mut r: int = 0;
    let mut sum: int = 0;
    let (tx, rx) = channel();
    tx.send(1);
    tx.send(2);
    tx.send(3);
    tx.send(4);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    tx.send(5);
    tx.send(6);
    tx.send(7);
    tx.send(8);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    r = rx.recv();
    sum += r;
    println!("{}", r);
    assert_eq!(sum, 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8);
}
