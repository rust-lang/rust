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
    let args = args.as_slice();
    if args.len() > 1 && args[1].as_slice() == "child" {
        if args[2].as_slice() == "green" {
            child();
        } else {
            let (tx, rx) = channel();
            native::task::spawn(proc() { tx.send(child()); });
            rx.recv();
        }
    } else {
        parent("green".to_string());
        parent("native".to_string());
        let (tx, rx) = channel();
        native::task::spawn(proc() {
            parent("green".to_string());
            parent("native".to_string());
            tx.send(());
        });
        rx.recv();
    }
}

fn parent(flavor: String) {
    let args = os::args();
    let args = args.as_slice();
    let mut p = io::process::Command::new(args[0].as_slice())
                                     .arg("child").arg(flavor).spawn().unwrap();
    p.stdin.get_mut_ref().write_str("test1\ntest2\ntest3").unwrap();
    let out = p.wait_with_output().unwrap();
    assert!(out.status.success());
    let s = str::from_utf8(out.output.as_slice()).unwrap();
    assert_eq!(s, "test1\n\ntest2\n\ntest3\n");
}

fn child() {
    for line in io::stdin().lines() {
        println!("{}", line.unwrap());
    }
}
