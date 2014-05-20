// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(phase)]
#[phase(syntax)] extern crate green;
green_start!(main)

fn start(n_tasks: int, token: int) {
    let (tx, mut rx) = channel();
    tx.send(token);
    for i in range(2, n_tasks + 1) {
        let (tx, next_rx) = channel();
        spawn(proc() roundtrip(i, tx, rx));
        rx = next_rx;
    }
    spawn(proc() roundtrip(1, tx, rx));
}

fn roundtrip(id: int, tx: Sender<int>, rx: Receiver<int>) {
    for token in rx.iter() {
        if token == 1 {
            println!("{}", id);
            break;
        }
        tx.send(token - 1);
    }
}

fn main() {
    let args = std::os::args();
    let args = args.as_slice();
    let token = if std::os::getenv("RUST_BENCH").is_some() {
        2000000
    } else {
        args.get(1).and_then(|arg| from_str(arg.as_slice())).unwrap_or(1000)
    };
    let n_tasks = args.get(2)
                      .and_then(|arg| from_str(arg.as_slice()))
                      .unwrap_or(503);

    start(n_tasks, token);
}
