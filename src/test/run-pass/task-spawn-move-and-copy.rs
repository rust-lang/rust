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

use std::thread::Thread;
use std::sync::mpsc::channel;

pub fn main() {
    let (tx, rx) = channel::<uint>();

    let x = box 1;
    let x_in_parent = &(*x) as *const int as uint;

    let _t = Thread::spawn(move || {
        let x_in_child = &(*x) as *const int as uint;
        tx.send(x_in_child).unwrap();
    });

    let x_in_child = rx.recv().unwrap();
    assert_eq!(x_in_parent, x_in_child);
}
