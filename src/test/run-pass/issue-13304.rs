// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast

extern crate green;
extern crate rustuv;
extern crate native;

use std::os;
use std::io;
use std::str;

#[start]
fn start(argc: int, argv: **u8) -> int {
    green::start(argc, argv, rustuv::event_loop, main)
}

fn main() {
    let args = os::args();
    if args.len() > 1 && args[1].as_slice() == "child" {
        if args[2].as_slice() == "green" {
            child();
        } else {
            let (tx, rx) = channel();
            native::task::spawn(proc() { tx.send(child()); });
            rx.recv();
        }
    } else {
        parent("green".to_owned());
        parent("native".to_owned());
        let (tx, rx) = channel();
        native::task::spawn(proc() {
            parent("green".to_owned());
            parent("native".to_owned());
            tx.send(());
        });
        rx.recv();
    }
}

fn parent(flavor: ~str) {
    let args = os::args();
    let mut p = io::Process::new(args[0].as_slice(), ["child".to_owned(), flavor]).unwrap();
    p.stdin.get_mut_ref().write_str("test1\ntest2\ntest3").unwrap();
    let out = p.wait_with_output();
    assert!(out.status.success());
    let s = str::from_utf8(out.output.as_slice()).unwrap();
    assert_eq!(s, "test1\n\ntest2\n\ntest3\n");
}

fn child() {
    for line in io::stdin().lines() {
        println!("{}", line.unwrap());
    }
}
