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

pub fn main() {
    let (tx, rx) = channel();

    // Spawn 10 tasks each sending us back one int.
    let mut i = 10;
    while (i > 0) {
        println!("{}", i);
        let tx = tx.clone();
        task::spawn({let i = i; proc() { child(i, &tx) }});
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    while (i > 0) {
        println!("{}", i);
        rx.recv();
        i = i - 1;
    }

    println!("main thread exiting");
}

fn child(x: int, tx: &Sender<int>) {
    println!("{}", x);
    tx.send(x);
}
