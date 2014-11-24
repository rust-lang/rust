// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate rustrt;

use std::io::process::{Command, ProcessOutput};
use std::os;
use std::str;
use std::rt;

use rustrt::unwind::try;

#[start]
fn start(argc: int, argv: *const *const u8) -> int {
    if argc > 1 {
        unsafe {
            match **argv.offset(1) {
                1 => {}
                2 => println!("foo"),
                3 => assert!(try(|| {}).is_ok()),
                4 => assert!(try(|| panic!()).is_err()),
                5 => assert!(try(|| spawn(proc() {})).is_err()),
                6 => assert!(Command::new("test").spawn().is_err()),
                _ => panic!()
            }
        }
        return 0
    }

    rt::start(argc, argv, main)
}

fn main() {
    let args = os::args();
    let me = args[0].as_slice();

    let x: &[u8] = &[1u8];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[2u8];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[3u8];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[4u8];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[5u8];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[6u8];
    pass(Command::new(me).arg(x).output().unwrap());
}

fn pass(output: ProcessOutput) {
    if !output.status.success() {
        println!("{}", str::from_utf8(output.output.as_slice()));
        println!("{}", str::from_utf8(output.error.as_slice()));
    }
}
