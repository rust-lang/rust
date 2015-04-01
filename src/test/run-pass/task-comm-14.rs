// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(std_misc)]

use std::sync::mpsc::{channel, Sender};
use std::thread;

pub fn main() {
    let (tx, rx) = channel();

    // Spawn 10 tasks each sending us back one isize.
    let mut i = 10;
    while (i > 0) {
        println!("{}", i);
        let tx = tx.clone();
        thread::scoped({let i = i; move|| { child(i, &tx) }});
        i = i - 1;
    }

    // Spawned tasks are likely killed before they get a chance to send
    // anything back, so we deadlock here.

    i = 10;
    while (i > 0) {
        println!("{}", i);
        rx.recv().unwrap();
        i = i - 1;
    }

    println!("main thread exiting");
}

fn child(x: isize, tx: &Sender<isize>) {
    println!("{}", x);
    tx.send(x).unwrap();
}
