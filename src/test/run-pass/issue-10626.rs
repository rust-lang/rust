// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

// Make sure that if a process doesn't have its stdio/stderr descriptors set up
// that we don't die in a large ball of fire

use std::env;
use std::process::{Command, Stdio};

pub fn main () {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        for _ in 0..1000 {
            println!("hello?");
        }
        for _ in 0..1000 {
            println!("hello?");
        }
        return;
    }

    let mut p = Command::new(&args[0]);
    p.arg("child").stdout(Stdio::null()).stderr(Stdio::null());
    println!("{:?}", p.spawn().unwrap().wait());
}
