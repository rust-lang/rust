// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi no processes
// ignore-emscripten no processes

use std::alloc::{Layout, handle_alloc_error};
use std::env;
use std::process::Command;
use std::str;

fn main() {
    if env::args().len() > 1 {
        handle_alloc_error(Layout::new::<[u8; 42]>())
    }

    let me = env::current_exe().unwrap();
    let output = Command::new(&me).arg("next").output().unwrap();
    assert!(!output.status.success(), "{:?} is a success", output.status);
    assert_eq!(str::from_utf8(&output.stderr).unwrap(), "memory allocation of 42 bytes failed");
}
