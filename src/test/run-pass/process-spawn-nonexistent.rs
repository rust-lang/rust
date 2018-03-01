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

use std::io::ErrorKind;
use std::process::Command;

fn main() {
    assert_eq!(Command::new("nonexistent")
                   .spawn()
                   .unwrap_err()
                   .kind(),
               ErrorKind::NotFound);
}
