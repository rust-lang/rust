// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(const_fn)]
#![allow(const_err)]

use std::env;
use std::process::{Command, Stdio};

const fn bar() -> usize { 0 - 1 }

fn foo() {
    let _: &'static _ = &bar();
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "test" {
        foo();
        return;
    }

    let mut p = Command::new(&args[0])
        .stdout(Stdio::piped())
        .stdin(Stdio::piped())
        .arg("test").output().unwrap();
    assert!(!p.status.success());
}