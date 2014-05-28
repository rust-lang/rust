// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use std::task;

fn start(tx: &Sender<Sender<String>>) {
    let (tx2, rx) = channel();
    tx.send(tx2);

    let mut a;
    let mut b;
    a = rx.recv();
    assert!(a == "A".to_string());
    println!("{}", a);
    b = rx.recv();
    assert!(b == "B".to_string());
    println!("{}", b);
}

pub fn main() {
    let (tx, rx) = channel();
    let _child = task::spawn(proc() { start(&tx) });

    let mut c = rx.recv();
    c.send("A".to_string());
    c.send("B".to_string());
    task::deschedule();
}
