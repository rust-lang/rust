// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// copyright 2013-2014 the rust project developers. see the copyright
// file at the top-level directory of this distribution and at
// http://rust-lang.org/copyright.
//
// licensed under the apache license, version 2.0 <license-apache or
// http://www.apache.org/licenses/license-2.0> or the mit license
// <license-mit or http://opensource.org/licenses/mit>, at your
// option. this file may not be copied, modified, or distributed
// except according to those terms.

// ignore-win32

use std::os;
use std::io::process::{Process, ExitSignal, ExitStatus};

pub fn main() {
    let args = os::args();
    let args = args.as_slice();
    if args.len() >= 2 && args[1] == "signal".to_owned() {
        // Raise a segfault.
        unsafe { *(0 as *mut int) = 0; }
    } else {
        let status = Process::status(args[0], ["signal".to_owned()]).unwrap();
        // Windows does not have signal, so we get exit status 0xC0000028 (STATUS_BAD_STACK).
        match status {
            ExitSignal(_) if cfg!(unix) => {},
            ExitStatus(0xC0000028) if cfg!(windows) => {},
            _ => fail!("invalid termination (was not signalled): {:?}", status)
        }
    }
}

