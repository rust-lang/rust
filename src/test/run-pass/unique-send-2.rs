// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::sync::mpsc::{channel, Sender};
use std::thread::Thread;

fn child(tx: &Sender<Box<uint>>, i: uint) {
    tx.send(box i).unwrap();
}

pub fn main() {
    let (tx, rx) = channel();
    let n = 100u;
    let mut expected = 0u;
    let _t = range(0u, n).map(|i| {
        expected += i;
        let tx = tx.clone();
        Thread::scoped(move|| {
            child(&tx, i)
        })
    }).collect::<Vec<_>>();

    let mut actual = 0u;
    for _ in range(0u, n) {
        let j = rx.recv().unwrap();
        actual += *j;
    }

    assert_eq!(expected, actual);
}
