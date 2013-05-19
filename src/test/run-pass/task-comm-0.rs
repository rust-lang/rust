// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

extern mod std;

use core::comm::Chan;
use core::comm::Port;

pub fn main() { test05(); }

fn test05_start(ch : &Chan<int>) {
    ch.send(10);
    error!("sent 10");
    ch.send(20);
    error!("sent 20");
    ch.send(30);
    error!("sent 30");
}

fn test05() {
    let (po, ch) = comm::stream();
    task::spawn(|| test05_start(&ch) );
    let mut value: int = po.recv();
    error!(value);
    value = po.recv();
    error!(value);
    value = po.recv();
    error!(value);
    assert_eq!(value, 30);
}
