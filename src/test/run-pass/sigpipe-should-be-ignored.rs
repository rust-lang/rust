// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Be sure that when a SIGPIPE would have been received that the entire process
// doesn't die in a ball of fire, but rather it's gracefully handled.

// ignore-aarch64
// ignore-emscripten

use std::env;
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

fn test() {
    let _ = io::stdin().read_line(&mut String::new());
    io::stdout().write(&[1]);
    assert!(io::stdout().flush().is_err());
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "test" {
        return test();
    }

    let mut p = Command::new(&args[0])
                        .stdout(Stdio::piped())
                        .stdin(Stdio::piped())
                        .arg("test").spawn().unwrap();
    drop(p.stdout.take());
    assert!(p.wait().unwrap().success());
}
