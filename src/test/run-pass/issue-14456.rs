// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-emscripten

#![feature(io, process_capture)]

use std::env;
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "child" {
        return child()
    }

    test();
}

fn child() {
    writeln!(&mut io::stdout(), "foo").unwrap();
    writeln!(&mut io::stderr(), "bar").unwrap();
    let mut stdin = io::stdin();
    let mut s = String::new();
    stdin.lock().read_line(&mut s).unwrap();
    assert_eq!(s.len(), 0);
}

fn test() {
    let args: Vec<String> = env::args().collect();
    let mut p = Command::new(&args[0]).arg("child")
                                     .stdin(Stdio::piped())
                                     .stdout(Stdio::piped())
                                     .stderr(Stdio::piped())
                                     .spawn().unwrap();
    assert!(p.wait().unwrap().success());
}
