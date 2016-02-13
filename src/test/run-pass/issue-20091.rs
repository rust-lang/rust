// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-aarch64
// ignore-emscripten
#![feature(std_misc, os)]

#[cfg(unix)]
fn main() {
    use std::process::Command;
    use std::env;
    use std::os::unix::prelude::*;
    use std::ffi::OsStr;

    if env::args().len() == 1 {
        assert!(Command::new(&env::current_exe().unwrap())
                        .arg(<OsStr as OsStrExt>::from_bytes(b"\xff"))
                        .status().unwrap().success())
    }
}

#[cfg(windows)]
fn main() {}
