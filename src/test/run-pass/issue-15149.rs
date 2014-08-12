// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(phase)]
extern crate native;
#[phase(plugin)]
extern crate green;

use native::NativeTaskBuilder;
use std::io::{TempDir, Command, fs};
use std::os;
use std::task::TaskBuilder;

green_start!(main)

fn main() {
    // If we're the child, make sure we were invoked correctly
    let args = os::args();
    if args.len() > 1 && args.get(1).as_slice() == "child" {
        return assert_eq!(args.get(0).as_slice(), "mytest");
    }

    test();
    let (tx, rx) = channel();
    TaskBuilder::new().native().spawn(proc() {
        tx.send(test());
    });
    rx.recv();
}

fn test() {
    // If we're the parent, copy our own binary to a tempr directory, and then
    // make it executable.
    let dir = TempDir::new("mytest").unwrap();
    let me = os::self_exe_name().unwrap();
    let dest = dir.path().join(format!("mytest{}", os::consts::EXE_SUFFIX));
    fs::copy(&me, &dest).unwrap();

    // Append the temp directory to our own PATH.
    let mut path = os::split_paths(os::getenv("PATH").unwrap_or(String::new()));
    path.push(dir.path().clone());
    let path = os::join_paths(path.as_slice()).unwrap();

    Command::new("mytest").env("PATH", path.as_slice())
                          .arg("child")
                          .spawn().unwrap();
}
