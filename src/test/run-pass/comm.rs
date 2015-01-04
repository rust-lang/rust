// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::thread::Thread;
use std::sync::mpsc::{channel, Sender};

pub fn main() {
    let (tx, rx) = channel();
    let _t = Thread::spawn(move|| { child(&tx) });
    let y = rx.recv().unwrap();
    println!("received");
    println!("{}", y);
    assert_eq!(y, 10);
}

fn child(c: &Sender<int>) {
    println!("sending");
    c.send(10).unwrap();
    println!("value sent");
}
