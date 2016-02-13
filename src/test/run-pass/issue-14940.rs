// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

use std::env;
use std::process::Command;
use std::io::{self, Write};

fn main() {
    let mut args = env::args();
    if args.len() > 1 {
        let mut out = io::stdout();
        out.write(&['a' as u8; 128 * 1024]).unwrap();
    } else {
        let out = Command::new(&args.next().unwrap()).arg("child").output();
        let out = out.unwrap();
        assert!(out.status.success());
    }
}
