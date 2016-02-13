// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
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
use std::process::{self, Command, Stdio};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        child();
    } else {
        parent();
    }
}

fn parent() {
    let args: Vec<String> = env::args().collect();
    let status = Command::new(&args[0]).arg("child").status().unwrap();
    assert_eq!(status.code(), Some(2));
}

fn child() -> i32 {
    process::exit(2);
}
