// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::task;

fn producer(tx: &Sender<Vec<u8>>) {
    tx.send(
         vec!(1u8, 2u8, 3u8, 4u8, 5u8, 6u8, 7u8, 8u8, 9u8, 10u8, 11u8, 12u8,
          13u8));
}

pub fn main() {
    let (tx, rx) = channel::<Vec<u8>>();
    let _prod = task::spawn(proc() {
        producer(&tx)
    });

    let _data: Vec<u8> = rx.recv();
}
