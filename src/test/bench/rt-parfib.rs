// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sync::mpsc::channel;
use std::env;
use std::thread;

// A simple implementation of parfib. One subtree is found in a new
// task and communicated over a oneshot pipe, the other is found
// locally. There is no sequential-mode threshold.

fn parfib(n: u64) -> u64 {
    if n == 0 || n == 1 {
        return 1;
    }

    let (tx, rx) = channel();
    thread::spawn(move|| {
        tx.send(parfib(n-1)).unwrap();
    });
    let m2 = parfib(n-2);
    return rx.recv().unwrap() + m2;
}

fn main() {
    let mut args = env::args();
    let n = if args.len() == 2 {
        args.nth(1).unwrap().parse::<u64>().unwrap()
    } else {
        10
    };

    parfib(n);

}
