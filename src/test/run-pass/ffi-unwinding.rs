// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unwinding_attributes)]

use std::io::Command;
use std::os;
use std::str;

struct Guard { name: &'static str }

impl Drop for Guard {
    fn drop(&mut self) {
        println!("Dropped Guard {}", self.name);
    }
}

fn main() {
    let args = os::args();

    if args.len() > 1 {
        if &args[1][] == "abort" {
            let _g = Guard { name: "abort" };

            i_abort();

            return;
        } else if &args[1][] == "unwind" {
            let _g = Guard { name: "unwind" };

            i_unwind();

            return;
        }
    }

    let p = Command::new(&args[0][])
        .arg("abort")
        .spawn().unwrap().wait_with_output().unwrap();

    assert!(!p.status.success());
    let mut lines = str::from_utf8(&p.output[]).unwrap().lines();

    assert!(lines.next().is_none());

    let p = Command::new(&args[0][])
        .arg("unwind")
        .spawn().unwrap().wait_with_output().unwrap();

    assert!(!p.status.success());
    let mut lines = str::from_utf8(&p.output[]).unwrap().lines();

    assert_eq!(&lines.next().unwrap()[], "Dropped Guard unwind");
}

#[inline(never)]
extern "C" fn i_abort() { panic!() }

#[inline(never)] #[can_unwind]
extern "C" fn i_unwind() { panic!() }
