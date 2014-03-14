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

fn start(tx: &Sender<int>, i0: int) {
    let mut i = i0;
    while i > 0 {
        tx.send(0);
        i = i - 1;
    }
}

pub fn main() {
    // Spawn a task that sends us back messages. The parent task
    // is likely to terminate before the child completes, so from
    // the child's point of view the receiver may die. We should
    // drop messages on the floor in this case, and not crash!
    let (tx, rx) = channel();
    task::spawn(proc() {
        start(&tx, 10)
    });
    rx.recv();
}
