// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten no threads support

#![feature(std_misc)]

use std::thread;
use std::sync::mpsc::{channel, Sender};

fn producer(tx: &Sender<Vec<u8>>) {
    tx.send(
         vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
          13]).unwrap();
}

pub fn main() {
    let (tx, rx) = channel::<Vec<u8>>();
    let prod = thread::spawn(move|| {
        producer(&tx)
    });

    let _data: Vec<u8> = rx.recv().unwrap();
    prod.join();
}
