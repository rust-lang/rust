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

extern crate extra;

use std::task;

fn start(tx: &Sender<int>, start: int, number_of_messages: int) {
    let mut i: int = 0;
    while i < number_of_messages { tx.send(start + i); i += 1; }
}

pub fn main() {
    info!("Check that we don't deadlock.");
    let (tx, rx) = channel();
    task::try(proc() { start(&tx, 0, 10) });
    info!("Joined task");
}
