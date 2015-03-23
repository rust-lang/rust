// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(start, os, std_misc, old_io)]

use std::ffi;
use std::old_io::process::{Command, ProcessOutput};
use std::os;
use std::rt::unwind::try;
use std::rt;
use std::str;
use std::thread::Thread;
use std::thunk::Thunk;

#[start]
fn start(argc: int, argv: *const *const u8) -> int {
    if argc > 1 {
        unsafe {
            match **argv.offset(1) {
                1 => {}
                2 => println!("foo"),
                3 => assert!(try(|| {}).is_ok()),
                4 => assert!(try(|| panic!()).is_err()),
                5 => assert!(Command::new("test").spawn().is_err()),
                _ => panic!()
            }
        }
        return 0
    }

    let args = unsafe {
        (0..argc as uint).map(|i| {
            let ptr = *argv.offset(i as int) as *const _;
            ffi::c_str_to_bytes(&ptr).to_vec()
        }).collect::<Vec<_>>()
    };
    let me = &*args[0];

    let x: &[u8] = &[1];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[2];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[3];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[4];
    pass(Command::new(me).arg(x).output().unwrap());
    let x: &[u8] = &[5];
    pass(Command::new(me).arg(x).output().unwrap());

    0
}

fn pass(output: ProcessOutput) {
    if !output.status.success() {
        println!("{:?}", str::from_utf8(&output.output));
        println!("{:?}", str::from_utf8(&output.error));
    }
}
