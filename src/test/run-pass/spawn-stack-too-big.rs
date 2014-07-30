// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-macos apparently gargantuan mmap requests are ok?

#![feature(phase)]

#[phase(plugin)]
extern crate green;
extern crate native;

use std::task::TaskBuilder;
use native::NativeTaskBuilder;

green_start!(main)

fn main() {
    test();

    let (tx, rx) = channel();
    TaskBuilder::new().native().spawn(proc() {
        tx.send(test());
    });
    rx.recv();
}

#[cfg(not(target_word_size = "64"))]
fn test() {}

#[cfg(target_word_size = "64")]
fn test() {
    let (tx, rx) = channel();
    spawn(proc() {
        TaskBuilder::new().stack_size(1024 * 1024 * 1024 * 64).spawn(proc() {
        });
        tx.send(());
    });

    assert!(rx.recv_opt().is_err());
}
