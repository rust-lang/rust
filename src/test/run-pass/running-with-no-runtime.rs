// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate native;

use std::io::process::{Command, ProcessOutput};
use std::os;
use std::str;
use std::rt::unwind::try;

local_data_key!(foo: int)

#[start]
fn start(argc: int, argv: **u8) -> int {
    if argc > 1 {
        unsafe {
            match **argv.offset(1) {
                1 => {}
                2 => println!("foo"),
                3 => assert!(try(|| {}).is_ok()),
                4 => assert!(try(|| fail!()).is_err()),
                5 => assert!(try(|| spawn(proc() {})).is_err()),
                6 => assert!(Command::new("test").spawn().is_err()),
                7 => assert!(foo.get().is_some()),
                8 => assert!(try(|| { foo.replace(Some(3)); }).is_err()),
                _ => fail!()
            }
        }
        return 0
    }

    native::start(argc, argv, main)
}

fn main() {
    let args = os::args();
    let me = args.get(0).as_slice();

    pass(Command::new(me).arg(&[1u8]).output().unwrap());
    pass(Command::new(me).arg(&[2u8]).output().unwrap());
    pass(Command::new(me).arg(&[3u8]).output().unwrap());
    pass(Command::new(me).arg(&[4u8]).output().unwrap());
    pass(Command::new(me).arg(&[5u8]).output().unwrap());
    pass(Command::new(me).arg(&[6u8]).output().unwrap());
}

fn pass(output: ProcessOutput) {
    if !output.status.success() {
        println!("{}", str::from_utf8(output.output.as_slice()));
        println!("{}", str::from_utf8(output.error.as_slice()));
    }
}
