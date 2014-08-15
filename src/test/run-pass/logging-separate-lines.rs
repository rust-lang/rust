// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-android
// ignore-windows
// exec-env:RUST_LOG=debug

#![feature(phase)]

#[phase(plugin, link)]
extern crate log;

use std::io::Command;
use std::os;
use std::str;

fn main() {
    let args = os::args();
    let args = args.as_slice();
    if args.len() > 1 && args[1].as_slice() == "child" {
        debug!("foo");
        debug!("bar");
        return
    }

    let p = Command::new(args[0].as_slice())
                    .arg("child")
                    .spawn().unwrap().wait_with_output().unwrap();
    assert!(p.status.success());
    let mut lines = str::from_utf8(p.error.as_slice()).unwrap().lines();
    assert!(lines.next().unwrap().contains("foo"));
    assert!(lines.next().unwrap().contains("bar"));
}
