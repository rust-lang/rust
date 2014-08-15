// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-windows

extern crate debug;

use std::os;
use std::io::process::{Command, ExitSignal, ExitStatus};

pub fn main() {
    let args = os::args();
    let args = args.as_slice();
    if args.len() >= 2 && args[1].as_slice() == "signal" {
        // Raise a segfault.
        unsafe { *(0 as *mut int) = 0; }
    } else {
        let status = Command::new(args[0].as_slice()).arg("signal").status().unwrap();
        // Windows does not have signal, so we get exit status 0xC0000028 (STATUS_BAD_STACK).
        match status {
            ExitSignal(_) if cfg!(unix) => {},
            ExitStatus(0xC0000028) if cfg!(windows) => {},
            _ => fail!("invalid termination (was not signalled): {:?}", status)
        }
    }
}
