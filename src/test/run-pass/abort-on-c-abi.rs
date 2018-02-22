// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Since we mark some ABIs as "nounwind" to LLVM, we must make sure that
// we never unwind through them.

// ignore-cloudabi no env and process
// ignore-emscripten no processes

#![feature(unwind_attributes)]

use std::{env, panic};
use std::io::prelude::*;
use std::io;
use std::process::{Command, Stdio};

#[unwind(aborts)]
extern "C" fn panic_in_ffi() {
    panic!("Test");
}

fn test() {
    let _ = panic::catch_unwind(|| { panic_in_ffi(); });
    // The process should have aborted by now.
    io::stdout().write(b"This should never be printed.\n");
    let _ = io::stdout().flush();
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
    assert!(!p.wait().unwrap().success());
}
